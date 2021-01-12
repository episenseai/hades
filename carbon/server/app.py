import os

import sanic.response as res
from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic_cors import CORS

from .config import server_config
from .routes import root_bp
from .routes.sse import check_sse_token
from .store import store_backend
from .utils import printBox

# from carbon.mlmodels.main import spawn_model_workers
# from carbon.mlpipeline.main import spawn_pipe_workers

# pipe_worker_process_kill = lambda: None
# model_worker_process_kill = lambda: None

app = Sanic("episense_backend_app")
app.blueprint(root_bp)

### CORS setting
CORS(
    app,
    resources={r"/*": {
        "origins": server_config.app.CORS_origins
    }},
    methods=server_config.app.CORS_methods,
    automatic_options=True,
    supports_credentials=True,
)


### Middleware to give blank response to CORS
@app.middleware("request")
async def cors_halt_request(request):
    if request.path != "/checks/health":
        if "origin" not in request.headers:
            return res.json({}, status=404)
        if request.headers["origin"] not in server_config.app.CORS_origins:
            return res.json({}, status=403)


# @app.middleware("response")
# async def cors_halt_response(request, response):
#     if ("origin" in request.headers) and (
#         request.headers["origin"] not in Conf.app.CORS
#     ):
#         return res.json({}, status=200)

### MIDDLEWARES for the server


@app.middleware("request")
async def authorization(request):
    # print(request.headers)
    # print("Authorization: ", request.token)
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
async def beforeStart(app, loop):
    try:
        os.mkdir(f"{server_config.jobs.uploads_folder}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{server_config.jobs.models_folder}")
    except FileExistsError:
        pass


# This is run after the server has successfully started
@app.listener("after_server_start")
async def notify_server_started(app, loop):
    printBox(f"Starting app - (episense ai) (ENV = {server_config.app.env})")
    # global pipe_worker_process_kill
    # pipe_worker_process_kill = spawn_pipe_workers()
    # global model_worker_process_kill
    # model_worker_process_kill = spawn_model_workers()


# This is run before the server starts shuttings down
@app.listener("before_server_stop")
async def notify_server_stopping(app, loop):
    # pipe_worker_process_kill()
    # model_worker_process_kill()
    printBox("Shutting down server....................................")


# This is run after all the remaining requests have been processesed
# @app.listener("after_server_stop")
# async def afterStop(app, loop):
#     printBox("Processed all remaining requests")


@app.exception(NotFound)
async def ignore_404s(request, exception):
    return res.json(
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
