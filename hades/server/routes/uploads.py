import os

from sanic import Blueprint, response

from ..env import env
from ..store import store_backend

uploads_bp = Blueprint("uploads_service", url_prefix="/uploads")


# route: /uploads/
@uploads_bp.post("/")
async def signup(request):
    try:
        if "userid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif ":" in request.files["file"][0].name:
            info = (
                "File name contains  ':'  , which is not allowed. Try again with a different name."
            )
            status = 400
        else:
            # print(request.files["file"][0].name)
            file_name = store_backend.timestamp_file_name(request.files["file"][0].name)
            folder_name = request.ctx.userid
            # save file to disk
            one_shot_upload(folder_name, file_name, request.files["file"][0].body)
            # save file name to redis
            store_backend.set_upload(request.ctx.userid, file_name)
            info = f"Successfully uploaded a new file named {request.files['file'][0].name}"
            status = 201
    except Exception as ex:
        info = ex.args[0]
        status = 500
    return response.json(
        {
            "success": True if (status == 201) else False,
            "version": "v1",
            "info": info,
            "data": {},
        },
        status=status,
    )


def one_shot_upload(folder_name, file_name, content):
    sink = f"{env().UPLOADS_VOLUME}/{folder_name}/{file_name}"
    # print("source ", sink)
    try:
        try:
            os.mkdir(f"{env().UPLOADS_VOLUME}/{folder_name}")
        except FileExistsError:
            pass

        with open(sink, mode="wb") as sinkfd:
            sinkfd.write(content)
    except Exception:
        # print(ex)
        if os.path.isfile(sink):
            os.remove(sink)
        raise


@uploads_bp.get("/")
async def list_uploads(request):
    data = {}
    try:
        if "userid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            data = store_backend.get_uploads(request.ctx.userid)
            info = "got list of uploaded files"
            status = 200
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
