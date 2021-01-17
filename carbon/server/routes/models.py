from typing import Optional

from pydantic.main import BaseModel
from sanic import Blueprint, response

from ..store import model_producer, pipe_producer, store_backend

models_bp = Blueprint("models_service", url_prefix="/tab/v1/models")


@models_bp.post("/build")
async def model_build(request):
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
                data = pipe_producer.current_pipe_state(request.ctx.userid, request.args["projectid"][0])
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
                        if "modelids" not in request.json:
                            modelids = []
                        else:
                            modelids = request.json["modelids"]
                        print(f"{modelids=}***********")

                        res, models_dict = model_producer.submit_model_jobs(
                            request.ctx.userid, request.args["projectid"][0], optimizeUsing, model_type, modelids
                        )
                        from devtools import debug

                        debug(res, models_dict)
                        # no model job was queued
                        if not res:
                            status = 400
                            # there were model job to be
                            if models_dict.models_to_build:
                                status = 500
                                info = f"Something fatal happened while submitting models jobs for {request.args['projectid'][0]}"
                            elif len(models_dict.models_rejected) == len(modelids):
                                info = "None of the modelids provided were relavant for this project. "
                            elif len(models_dict.models_to_ignore) == len(modelids):
                                info = "None of the modelids have status in [DONE, ERROR, CANCELLED]"
                            elif (len(models_dict.models_rejected) + len(models_dict.models_to_ignore)) == len(
                                modelids
                            ):
                                info = (
                                    "Some of the modelids were irrelavant for the project "
                                    + "and the rest of them do not have status in [DONE, ERROR, CANCELLED]."
                                )
                            else:
                                status = 500
                                info = "Something unknown happended."
                        else:
                            data = pipe_producer.current_pipe_state(request.ctx.userid, request.args["projectid"][0])
                            if data is None:
                                info = (
                                    "Submitted jobs, but couldn't get current state of the pipeline."
                                    + "Try refreshing the page"
                                )
                                status = 500
                            else:
                                data["id"] = request.args["projectid"][0]
                                data["name"] = proj[0]
                                data["timestamp"] = proj[1]
                                info = f"Successfully submitted {model_type} model jobs for the {proj[0]} project"
                                status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    finally:
        from devtools import debug

        debug(models_dict.dict())
        return response.json(
            {
                "success": True if (status == 200) else False,
                "version": "v1",
                "info": info,
                "data": data if (status == 200) else {},
                "models_dict": models_dict.dict(),
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
                models_list = model_producer.get_models_list(request.ctx.userid, request.args["projectid"][0])
                if not models_list:
                    info = f"There are no models for this project"
                    status = 400
                else:
                    data = model_producer.get_model_data(request.ctx.userid, request.args["projectid"][0], models_list)
                    # pprint(data)
                    info = f"Got models data for the {proj[0]} project"
                    status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    finally:
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
                )
                # pprint(data)
                info = f"Got models data for the {proj[0]} project"
                status = 200
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    finally:
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
                data = model_producer.cancel_job(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    request.json["modelid"],
                )
                # pprint(data)
                if data:
                    status = 200
                    if data[0] == 0:
                        info = "Model is not in the job queue."
                    elif data[0] == 1:
                        info = "Model build was already cancelled from a previous request."
                    else:
                        info = "Model was successfully marked for cancellation."
                else:
                    info = "Error trying to cancel the model build"
                    status = 500
                from devtools import debug

                debug(data)
    except Exception as ex:
        import traceback

        print(traceback.format_exc())
        info = ex.args[0]
        status = 500
    finally:
        return response.json(
            {
                "success": True if (status == 200) else False,
                "version": "v1",
                "info": info,
                "data": {},
            },
            status=status,
        )
