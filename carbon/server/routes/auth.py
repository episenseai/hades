from sanic import Blueprint, response

from ..store import store_backend

auth_bp = Blueprint("authentication_service", url_prefix="/auth")


# route: /auth/signup
@auth_bp.post("/signup")
async def signup(request):
    userid = None
    try:
        if "username" not in request.json or "password" not in request.json:
            info = "Bad request. missing parameters"
            status = 400
        else:
            userid = store_backend.add_user(request.json["username"], request.json["password"])
            status = 200
            if userid is None:
                info = "Username already exits. Try a different one."
            else:
                info = f"Successfull signup for username: {request.json['username']}"
    except Exception as ex:
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 200 and userid is not None) else False,
            "version": "v1",
            "info": info,
            "data": {},
        },
        status=status,
    )


# route: /auth/login
@auth_bp.post("/login")
async def login(request):
    jwt = None
    try:
        if "username" not in request.json or "password" not in request.json:
            info = "Bad request. missing parameters"
            status = 400
        else:
            jwt = store_backend.issue_jwt(request.json["username"], request.json["password"])
            status = 200
            if jwt is None:
                info = "Either username or password is wrong. New user? Signup to create an account and then try logging in."
            else:
                info = f"Successfully logged in as {request.json['username']}"
    except Exception as ex:
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 200 and jwt is not None) else False,
            "version": "v1",
            "info": info,
            "data": {"jwt": jwt} if (status == 200 and jwt is not None) else {},
        },
        status=status,
    )


# route: /auth/logout
@auth_bp.post("/logout")
async def logout(_request):
    # currently does nothing
    return response.json(
        {
            "success": True,
            "version": "v1",
            "info": "Successfully logged out.",
            "data": {},
        },
        status=201,
    )
