from sanic import Blueprint, response

from ..store import pipe_producer, store_backend

pipe_bp = Blueprint("pipe_service", url_prefix="/tab/v1/pipe")


@pipe_bp.post("/next")
async def next_stage(request):
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
            elif (
                "current_stage" not in request.json
                or "next_stage" not in request.json
                or "data" not in request.json
            ):
                info = "Bad request. missing values from post data"
                status = 400
            elif (
                (request.json["current_stage"] not in seq[:-1])
                or (request.json["next_stage"] not in seq[:-1])
                or (request.json["current_stage"].split(":")[1] != "GET")
                or (request.json["next_stage"].split(":")[1] != "POST")
                or (
                    (seq.index(request.json["current_stage"]) + 1)
                    != seq.index(request.json["next_stage"])
                )
            ):
                info = "Bad request. wrong data in post request"
                status = 400
            else:
                res = pipe_producer.submit_job(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    request.json["next_stage"],
                    request.json["data"],
                )
                if res:
                    data = pipe_producer.current_pipe_state(
                        request.ctx.userid, request.args["projectid"][0]
                    )
                    if data is None:
                        info = "submitted the job but something happened while getting current pipe status"
                        status = 500
                    else:
                        data["id"] = request.args["projectid"][0]
                        data["name"] = proj[0]
                        data["timestamp"] = proj[1]
                        info = f"successfully submitted the job for {request.json['next_stage']}"
                        status = 200
                else:
                    info = "somehting fatal happened while submitting the job"
                    status = 500
    except Exception as ex:
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


@pipe_bp.post("/current")
async def stage_data(request):
    data = {}
    try:
        if "userid" not in request.args and "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif (
            store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0]) is None
        ):
            info = f"Projectid {request.args['projectid'][0]} not associated with the user"
            status = 400
        elif "current_stage" not in request.json:
            info = "Bad request. missing current_stage value."
            status = 400
        elif request.json["current_stage"] not in seq:
            info = "Bad request. current_stage value is not a valid stage."
            status = 400
        else:
            data = pipe_producer.current_stage_data(
                request.ctx.userid,
                request.args["projectid"][0],
                request.json["current_stage"],
            )
            if data is None:
                info = "Stage requested is not the current stage."
                status = 400
            else:
                info = f"got stage data for {request.json['current_stage']}"
                status = 200

    except Exception as ex:
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


seq = [
    "consume:GET",
    "consume:POST",
    "prepare:GET",
    "prepare:POST",
    "transform:GET",
    "transform:POST",
    "build:GET",
    "build:POST",
    "finalconfig:GET",
]


@pipe_bp.post("/revert")
async def revert_stage(request):
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
            elif "from" not in request.json or "to" not in request.json:
                info = "Bad request data."
                status = 400
            elif request.json["from"] not in seq or request.json["to"] not in seq:
                info = f"Bad stage names in data from = {request.json['from']} to = {request.json['to']}"
                status = 400
            elif seq[seq.index(request.json["to"])].split(":")[1] != "GET":
                info = f"Can not revert from {request.json['from']} to {request.json['to']}. Can only revert to GET stages."
                status = 400
            elif seq.index(request.json["to"]) >= seq.index(request.json["from"]):
                info = f"Can not revert from {request.json['from']} to {request.json['to']}. Only reversals to previous stages are allowed."
                status = 400
            else:
                res = pipe_producer.revert_stage(
                    request.ctx.userid,
                    request.args["projectid"][0],
                    request.json["from"],
                    request.json["to"],
                )
                if res:
                    data = pipe_producer.current_pipe_state(
                        request.ctx.userid, request.args["projectid"][0]
                    )
                    if data is None:
                        info = "successfully reverted to a previous stage but somehting fatal happened while getting current pipe state"
                        status = 500
                    else:
                        data["id"] = request.args["projectid"][0]
                        data["name"] = proj[0]
                        data["timestamp"] = proj[1]
                        info = f"successfully reverted to {request.json['to']} stage."
                        status = 200
                else:
                    info = "somehting fatal happened reverting to a previous stage"
                    status = 500

    except Exception as ex:
        # import traceback

        # print(traceback.format_exc())
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


@pipe_bp.post("/unfreeze")
async def unfreeze_pipe(request):
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
            elif pipe_producer.unfreeze_pipe(request.ctx.userid, request.args["projectid"][0]):
                data = pipe_producer.current_pipe_state(
                    request.ctx.userid, request.args["projectid"][0]
                )
                if data is None:
                    info = "successfully unfreezed the pipe but something fatal happened while getting  current pipe state"
                    status = 500
                else:
                    data["id"] = request.args["projectid"][0]
                    data["name"] = proj[0]
                    data["timestamp"] = proj[1]
                    info = (
                        f"successfully reset error in the pipe for {request.args['projectid'][0]}"
                    )
                    status = 200
            else:
                info = "Bad request. Can not unfreeze pipe."
                status = 400

    except Exception as ex:

        info = ex.args[0]
        status = 500
        # import traceback

        # print(traceback.format_exc())
    return response.json(
        {
            "success": True if (status == 200) else False,
            "version": "v1",
            "info": info,
            "data": data if (status == 200) else {},
        },
        status=status,
    )
