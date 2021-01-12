from sanic import Blueprint, response
from carbon.redis_task import main_app, pipe_producer, model_producer

models_bp = Blueprint("models_service", url_prefix="/tab/v1/models")


@models_bp.post("/build")
async def model_build(request):
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            proj = main_app.verify_projectid(request.ctx.userid, request.args["projectid"][0])
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
                elif data["pipe_status"] != "0":
                    info = f"Bad request. Duplicate request to submit model jobs."
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
                        data = model_producer.submit_model_jobs(
                            request.ctx.userid,
                            request.args["projectid"][0],
                            model_type,
                            optimizeUsing,
                        )
                        if data is None:
                            info = f"Something fatal happened while submitting models jobs for {request.args['projectid'][0]}"
                            status = 500
                        else:
                            data = pipe_producer.current_pipe_state(request.ctx.userid,
                                                                    request.args["projectid"][0])
                            if data is None:
                                info = f"Submitted model jobs, but couldn't get current state of the pipeline. Try refreshing the page"
                                status = 500
                            else:
                                data["id"] = request.args["projectid"][0]
                                data["name"] = proj[0]
                                data["timestamp"] = proj[1]
                                info = f"Successfully submitted {model_type} model jobs for the {proj[0]} project"
                                status = 200
    except Exception as ex:
        # import traceback

        # print(traceback.format_exc())
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


@models_bp.get("/")
async def model_results(request):
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            proj = main_app.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            else:
                models_list = model_producer.get_models_list(request.ctx.userid, request.args["projectid"][0])
                if not models_list:
                    info = f"There are no models for this project"
                    status = 400
                else:
                    data = model_producer.get_model_data(request.ctx.userid, request.args["projectid"][0],
                                                         models_list)
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
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif "modelid" not in request.json:
            info = "Bad request. missing 'modelid' in json body"
            status = 400
        else:
            proj = main_app.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            elif request.json["modelid"] not in model_producer.get_models_list(
                    request.ctx.userid, request.args["projectid"][0]):
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
