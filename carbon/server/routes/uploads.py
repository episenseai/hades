from sanic import Blueprint, response
import os
from ..config import server_config
from carbon.redis_task import main_app

uploads_bp = Blueprint("uploads_service", url_prefix="/uploads")


# route: /uploads/
@uploads_bp.post("/")
async def signup(request):
    try:
        if "userid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        elif not main_app.userid_exists(request.ctx.userid):
            info = "Unauthorized request. Userid does not exist."
            status = 401
        elif ":" in request.files["file"][0].name:
            info = "File name contains  ':'  , which is not allowed. Try again with a different name."
            status = 400
        else:
            # print(request.files["file"][0].name)
            file_name = main_app.timestamp_file_name(request.files["file"][0].name)
            folder_name = request.ctx.userid
            # save file to disk
            one_shot_upload(folder_name, file_name, request.files["file"][0].body)
            # save file name to redis
            main_app.set_upload(request.ctx.userid, file_name)
            info = f"Successfully uploaded a new file named {request.files['file'][0].name}"
            status = 201
    except Exception as ex:
        info = ex.args[0]
        status = 500
    finally:
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
    import pathlib

    sink = f"{server_config.jobs.uploads_folder}/{folder_name}/{file_name}"
    # print("source ", sink)
    try:
        try:
            os.mkdir(f"{server_config.jobs.uploads_folder}/{folder_name}")
        except FileExistsError:
            pass

        with open(sink, mode="wb") as sinkfd:
            sinkfd.write(content)
    except Exception as ex:
        # print(ex)
        if os.path.isfile(sink):
            os.remove(sink)
        raise


@uploads_bp.get("/")
async def list_uploads(request):
    try:
        if "userid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            data = main_app.get_uploads(request.ctx.userid)
            info = "got list of uploaded files"
            status = 200
    except Exception as ex:
        info = ex.args[0]
        status = 500
        # import traceback

        # print(traceback.format_exc())
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
