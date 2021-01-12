import asyncio
import base64
import uuid
from concurrent.futures import CancelledError

from sanic import Blueprint, response
from sanic.response import stream

from carbon.redis_task import main_app, model_producer, pipe_producer

sse_bp = Blueprint("see_events_stream", url_prefix="/sse")


# TODO: cancel the loop when the server is stopped
@sse_bp.route("/events")
async def sse(request):
    userid = request.args["userid"][0]
    projectid = request.args["projectid"][0]
    proj = request.ctx.proj

    async def sample_streaming_fn(response):
        try:
            # i = 1
            while True:
                await asyncio.sleep(3)
                data = pipe_producer.current_pipe_state(userid, projectid)
                if data is None:
                    raise CancelledError("could not get current pipe state")
                else:
                    data["id"] = projectid
                    data["name"] = proj[0]
                    data["timestamp"] = proj[1]
                data = pipe_producer.to_JSON(data)
                data = base64.b64encode(data.encode()).decode()
                s = "data: " + str(data) + "\r\n\r\n"
                # print(s)
                # s = "data: " + str(i) + "hello sse" + "\r\n\r\n"
                # print(s.encode())
                # print(data)
                await response.write(s.encode())
                # i += 1
        except CancelledError as ex:
            pass
            # import traceback

            # print(traceback.format_exc())
            # print(ex)
        except Exception as ex:
            # import traceback

            # print(traceback.format_exc())
            print(ex)

    return stream(sample_streaming_fn, content_type="text/event-stream")


# TODO: cancel the loop when the server is stopped
@sse_bp.route("/events/models")
async def sse_models(request):
    userid = request.args["userid"][0]
    projectid = request.args["projectid"][0]
    proj = request.ctx.proj

    async def sample_streaming_fn(response):
        try:
            # i = 1
            while True:
                await asyncio.sleep(3)
                data = model_producer.get_model_status(userid, projectid)
                if not data:
                    raise CancelledError("could not get current model status")
                data = pipe_producer.to_JSON({"projectid": projectid, "data": data})
                data = base64.b64encode(data.encode()).decode()
                s = "data: " + str(data) + "\r\n\r\n"
                # print(s)
                # s = "data: " + str(i) + "hello sse" + "\r\n\r\n"
                # print(s.encode())
                # print(data)
                await response.write(s.encode())
                # i += 1
        except CancelledError as ex:
            pass
            # import traceback

            # print(traceback.format_exc())
            print(ex)
        except Exception as ex:
            import traceback

            print(traceback.format_exc())
            print(ex)

    return stream(sample_streaming_fn, content_type="text/event-stream")


one_shot_tokens = set()


@sse_bp.get("/token")
async def sse_token(request):
    try:
        if "userid" not in request.args or "projectid" not in request.args:
            info = "Bad request for getting SSE tokens. missing parameters"
            status = 400
        else:
            proj = main_app.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            else:
                token = uuid.uuid4().hex
                one_shot_tokens.add(token)
                data = {"token": token}
                info = "generated token for getting job updates from the server."
                status = 200
    except Exception as ex:
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


def check_sse_token(request):
    try:
        if ("userid" not in request.args or "projectid" not in request.args or "token" not in request.args):
            # print("bad params")
            return False
        else:
            proj = main_app.verify_projectid(request.args["userid"][0], request.args["projectid"][0])
            if proj is None:
                # print("project unknown")
                return False
            elif request.args["token"][0] not in one_shot_tokens:
                # print("invalid token")
                return False
            else:
                one_shot_tokens.discard(request.args["token"][0])
                return proj
    except Exception as ex:
        # import traceback

        # print(traceback.format_exc())
        print(ex)
        return False
