import asyncio
import os

from sanic import Sanic, response
from sanic.exceptions import NotFound
from sanic_cors import CORS

from .auth.validate import validate_token
from .env import Env, env
from .routes import root_bp
from .routes.sse import check_sse_token

app = Sanic("hades")
app.blueprint(root_bp)

# CORS setting
CORS(
    app,
    origins=[env().CORS_ORIGIN] if env().CORS_ENABLED else [],
    methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"]
    if env().CORS_ENABLED
    else [],
    automatic_options=True,
    supports_credentials=True,
)


@app.middleware("request")
async def authorization(request):
    """
    OAUTH token validation middleware for the server
    """
    # debug(request.headers)
    # print("path: ", request.path)
    # print(request.args)

    request.ctx.env = env().ENV.value

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

    else:
        is_expired = None
        if request.token is not None:
            (decoded_token, is_expired) = await validate_token(request.token)
        if request.token is None or decoded_token is None:
            # NOTE: auth tarpit: introduce a delay in response when the validation fails.
            # This is should be done in a more proper way.
            # Sleep for 5 secnonds
            if env().ENV == Env.PRODUCTION:
                await asyncio.sleep(5)

            headers = {"WWW-Authenticate": "Bearer"}
            if is_expired:
                headers["Access-Control-Expose-Headers"] = "X-Expired-AccessToken"
                headers["X-Expired-AccessToken"] = "1"
            return response.json(
                {
                    "success": False,
                    "version": "v1",
                    "info": "Unauthorized Request. Expired or invalid credentials",
                },
                status=401,
                headers=headers,
            )
        else:
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
