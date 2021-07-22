import os
import asyncio

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic_cors import CORS

from .auth.validate import validate_token
from .env import env
from .routes import root_bp
from .routes.sse import check_sse_token
from .store import store_backend

app = Sanic("hades")
app.blueprint(root_bp)

# CORS setting
CORS(
    app,
    resources={r"/*": {}},
    methods=env().cors_methods,
    automatic_options=True,
    supports_credentials=True,
)


@app.middleware("request")
async def cors_halt_request(request):
    pass


@app.middleware("request")
async def authorization(request):
    """
    OAUTH token validation middleware for the server
    """
    # debug(request.headers)
    # print("path: ", request.path)
    # print(request.args)

    if request.path in ["/sse/events", "/sse/events/models"]:
        proj = check_sse_token(request)
        if not proj:
            return response.json(
                {
                    "success": False,
                    "version": "v1",
                    "info": "Unauthorized Request. Can not verify SSE token or projectid",
                    "data": {},
                },
                status=401,
            )
        else:
            request.ctx.proj = proj

    if request.path not in [
        "/auth/login",
        "/auth/signup",
        "/sse/events",
        "/sse/events/models",
        "/checks/health",
    ]:
        if request.token is None:
            return response.json(
                {
                    "success": False,
                    "version": "v1",
                    "info": "Unauthorized Request. Please Login to continue...",
                },
                status=401,
            )
        else:
            decoded_token = await validate_token(request.token)
            if decoded_token is None:
                # NOTE: auth tarpit: introduce a delay in response when the validation fails.
                # This is should be done in a more proper way.
                # Sleep for 5 secnonds
                await asyncio.sleep(5)

                return response.json(
                    {
                        "success": False,
                        "version": "v1",
                        "info": "Unauthorized Request. Expired or invalid credentials",
                    },
                    status=401,
                )

            userid = str(decoded_token.sub)
            if "userid" in request.args:
                if len(request.args["userid"]) != 1:
                    return response.json(
                        {
                            "success": False,
                            "version": "v1",
                            "info": "Bad request query params",
                        },
                        status=400,
                    )
                if request.args["userid"][0] != userid:
                    return response.json(
                        {
                            "success": False,
                            "version": "v1",
                            "info": "Unauthorized request. Token issued for a different userid",
                        },
                        status=401,
                    )

            request.ctx.userid = userid


# This is run before the server starts accepting connections
@app.listener("before_server_start")
async def beforeStart(_app, _loop):
    try:
        os.mkdir(f"{env().UPLOADS_VOLUME}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{env().MODELS_VOLUME}")
    except FileExistsError:
        pass


# This is run after the server has successfully started
@app.listener("after_server_start")
async def notify_server_started(_app, _loop):
    print("============ STARTING (SERVER) ============")


# This is run before the server starts shuttings down
@app.listener("before_server_stop")
async def notify_server_stopping(_app, _loop):
    print("============ STOPING  (SERVER) ============")


# This is run after all the remaining requests have been processesed
# @app.listener("after_server_stop")
# async def afterStop(app, loop):
#     printBox("Processed all remaining requests")


@app.exception(NotFound)
async def ignore_404s(request, _exception):
    return response.json(
        {
            "success": False,
            "version": "v1",
            "info": "Yep, I totally found the page: {}".format(request.url),
        },
        status=404,
    )
