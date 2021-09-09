from typing import Optional

from pydantic.main import BaseModel
from sanic import Blueprint, response
from ..env import env, Env
from ..store import metrics_db, model_producer, pipe_producer, store_backend

models_bp = Blueprint("models_service", url_prefix="/tab/v1/models")


@models_bp.post("/build")
async def model_build(request):
    quota_error = False
    models_dict: Optional[BaseModel] = None
    data = []
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            proj = store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            else:
                data = pipe_producer.current_pipe_state(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    include_error=(request.ctx.env == "DEV"),
                )
                if data is None:
                    info = f"Something unexpected happened while getting the model state of {request.args['projectid'][0]} for submitting model build jobs"
                    status = 500
                elif data["current_stage"] != "finalconfig:GET":
                    info = f"Bad request. To submit model build jobs pipeline should be at finalconfig:GET stage but current stage of the pipeline is {data['current_stage']}"
                    status = 400
                elif data["pipe_status"] == "-1":
                    info = f"Bad request. Pipeline is freezed at {data['current_stage']} due to an error."
                    status = 400
                else:
                    data = pipe_producer.get_stage_data(
                        request.ctx.userid,
                        request.args["projectid"][0],
                        "finalconfig:GET",
                    )
                    if data is None:
                        info = f"Something unexpected happened while getting the curent stage data for {request.args['projectid'][0]}"
                        status = 500
                    else:
                        model_type = data["data"]["model_type"]
                        optimizeUsing = data["data"]["optimizeUsing"]

                        init = request.json.get("init", False) is True
                        if (
                            "modelids" not in request.json
                            or not request.json["modelids"]
                            or not isinstance(request.json["modelids"], list)
                        ):
                            modelids = []
                        else:
                            modelids = list(set(request.json["modelids"]))

                        state_tally = model_producer.model_state_tally(
                            request.ctx.userid, request.args["projectid"][0], model_type
                        )
                        state_quota = 0
                        if state_tally is not None:
                            for s in ["WAIT", "RUNNING", "TRYCANCEL"]:
                                state_quota += state_tally.get(s, 0)

                        quota = metrics_db.models_build(request.ctx.userid, requested=len(modelids))
                        if len(modelids) == 0 and init is False:
                            status = 400
                            info = "No models provided for build"
                        elif quota is None:
                            quota_error = True
                            status = 400
                            info = "Invalid request for model building"
                        elif quota[0] is False and init is False:
                            quota_error = True
                            status = 400
                            info = "Build Limit Exceeded: Try after some time."
                        elif env().ENV != Env.DEV and state_tally is not None and state_quota >= 3:
                            quota_error = True
                            status = 400
                            info = "Queue Limit Exceeded: wait for some models to finish or clear the queue."
                        else:
                            if quota[0] is False and init is True:
                                print("INFO: quota exceeded: we will only initialize the pipeline")
                                modelids = []
                            changed_hparams = request.json.get("changed_hparams", {})
                            res, models_dict = model_producer.submit_model_jobs(
                                request.ctx.userid,
                                request.args["projectid"][0],
                                optimizeUsing,
                                model_type,
                                modelids,
                                changed_hparams=changed_hparams,
                                init=init,
                            )

                            # no model job was queued
                            if not res and not init:
                                status = 400
                                # there were model job to be
                                if len(models_dict.models_rejected) > 0:
                                    info = "Invalid Models IDs"
                                elif len(models_dict.models_to_build) > 0:
                                    print(
                                        f"Something fatal happened while submitting models jobs for {request.args['projectid'][0]}"
                                    )
                                    info = "Invalid request for model building"
                                else:
                                    info = "Invalid request for model building"
                            else:
                                data = pipe_producer.current_pipe_state(
                                    request.ctx.userid,
                                    request.args["projectid"][0],
                                    include_error=(request.ctx.env == "DEV"),
                                )
                                if data is None:
                                    info = (
                                        "Submitted jobs but unable to get current status."
                                        + "Try refreshing the page"
                                    )
                                    status = 400
                                else:
                                    data["id"] = request.args["projectid"][0]
                                    data["name"] = proj[0]
                                    data["timestamp"] = proj[1]
                                    if init:
                                        info = "Successfully initialized the pipeline. You can start building models."
                                    else:
                                        info = (
                                            f"Successfully submitted jobs for the {proj[0]} project"
                                        )
                                    status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 200) else False,
            "version": "v1",
            "info": info,
            "data": data if (status == 200) else {},
            "models_dict": models_dict.dict() if models_dict else {},
            "quota_error": quota_error,
        },
        status=status,
    )


@models_bp.get("/")
async def model_results(request):
    data = {}
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            proj = store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            else:
                models_list = model_producer.get_models_list(
                    request.ctx.userid, request.args["projectid"][0]
                )
                if not models_list:
                    info = "There are no models for this project"
                    status = 400
                else:
                    data = model_producer.get_model_data(
                        request.ctx.userid,
                        request.args["projectid"][0],
                        models_list,
                        include_error=(request.ctx.env == "DEV"),
                    )
                    # debug(data)
                    info = f"Got models data for the {proj[0]} project"
                    status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 200) else False,
            "version": "v1",
            "info": info,
            "data": data if (status == 200) else {},
        },
        status=status,
    )


@models_bp.post("/")
async def get_model_result(request):
    data = {}
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif "modelid" not in request.json:
            info = "Bad request. missing 'modelid' in json body"
            status = 400
        else:
            proj = store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            elif request.json["modelid"] not in model_producer.get_models_list(
                request.ctx.userid, request.args["projectid"][0]
            ):
                info = "Bad reuqest. modelid not associated with the project"
                status = 400
            else:
                data = model_producer.get_model_data(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    [request.json["modelid"]],
                    include_error=(request.ctx.env == "DEV"),
                )
                # pprint(data)
                info = f"Got models data for the {proj[0]} project"
                status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 200) else False,
            "version": "v1",
            "info": info,
            "data": data["models"][0] if (status == 200) else {},
        },
        status=status,
    )


@models_bp.post("/cancel")
async def cancel_model_job(request):
    print("Cancellation request")
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif "modelid" not in request.json or not isinstance(request.json["modelid"], str):
            info = "Invalid request"
            status = 400
        else:
            proj = store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            elif request.json["modelid"] not in model_producer.get_models_list(
                request.ctx.userid, request.args["projectid"][0]
            ):
                info = "Bad reuqest. modelid not associated with the project"
                status = 400
            else:
                data = model_producer.cancel_job(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    request.json["modelid"],
                )
                if data:
                    status = 200
                    if data[0] == 0:
                        info = "Model is not in the job queue."
                    elif data[0] == 1:
                        info = "Model build was already cancelled from a previous request."
                    else:
                        if data[1] and data[1][0] == "WAIT":
                            if not metrics_db.models_cancel(request.ctx.userid, 1):
                                print("ERROR: could not add cancelled model event")
                        info = "Model was successfully marked for cancellation."
                else:
                    info = "Error trying to cancel the model build"
                    status = 400
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 400
    return response.json(
        {
            "success": True if (status == 200) else False,
            "version": "v1",
            "info": info,
            "data": {},
        },
        status=status,
    )
