import os

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic_cors import CORS

from .env import env
from .routes import root_bp
from .routes.sse import check_sse_token
from .store import store_backend

app = Sanic("episense_backend_app")
app.blueprint(root_bp)

### CORS setting
CORS(
    app,
    resources={r"/*": {}},
    methods=env().cors_methods,
    automatic_options=True,
    supports_credentials=True,
)


### Middleware to give blank response to CORS
@app.middleware("request")
async def cors_halt_request(request):
    print(env().cors_origins)
    print(request.headers)
    if request.path != "/checks/health":
        if "origin" not in request.headers:
            return response.json({}, status=404)
        if False and request.headers["origin"] not in env().cors_origins:
            return response.json({}, status=403)


### MIDDLEWARES for the server
@app.middleware("request")
async def authorization(request):
    print(request.headers)
    print("Authorization: ", request.token)
    print("path: ", request.path)
    print(request.args)

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
            decoded_token = store_backend.verify_jwt(request.token)
            if decoded_token is None:
                return response.json(
                    {
                        "success": False,
                        "version": "v1",
                        "info": "Unauthorized Request. Expired or invalid credentials",
                    },
                    status=401,
                )
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
                if request.args["userid"][0] != decoded_token["userid"]:
                    return response.json(
                        {
                            "success": False,
                            "version": "v1",
                            "info": "Unauthorized request. Token issued for a different userid",
                        },
                        status=401,
                    )
            request.ctx.username = decoded_token["username"]
            request.ctx.userid = decoded_token["userid"]


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


###  serve STATIC files
# Some browsers request it by default (prevent unnecessary 404 on server)
app.static("/favicon.ico", "./static/favicon.ico", name="favicon", content_type="image/x-icon")
