from sanic import Blueprint, response

from ..store import pipe_producer, store_backend

projects_bp = Blueprint("projects_service", url_prefix="/tab/v1/projects")


# route: /projects?userid=26r276re25r52456
@projects_bp.post("/")
async def create_project(request):
    try:
        if ("projectname" not in request.json or "projectdesc" not in request.json or
                "userid" not in request.args):
            info = "Bad request. missing parameters"
            status = 400
        elif ":" in request.json["projectname"]:
            info = "Project name contains  ':'  , which is not allowed. Try again with a different name."
            status = 400
        else:
            projectid = store_backend.add_project(
                request.ctx.userid,
                request.json["projectname"],
                request.json["projectdesc"],
            )
            pipe_producer.setup_new_pipe(request.ctx.userid, projectid)
            status = 200
            info = f"Successfully created a new project named: {request.json['projectname']}"
    except Exception as ex:
        info = ex.args[0]
        status = 500
    finally:
        return response.json(
            {
                "success": True if (status == 200) else False,
                "version": "v1",
                "info": info,
                "data": {},
            },
            status=status,
        )


@projects_bp.get("/")
async def list_projects(request):
    try:
        if "userid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            data = store_backend.projects_list(request.ctx.userid)
            info = "list of projects"
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


@projects_bp.post("/current")
async def set_project(request):
    try:
        if "userid" not in request.args or "projectid" not in request.args:
            info = "Bad request. missing parameters"
            status = 400
        else:
            proj = store_backend.verify_projectid(request.ctx.userid, request.args["projectid"][0])
            if proj is None:
                info = f"Projectid {request.args['projectid'][0]} not associated with the user"
                status = 400
            else:
                store_backend.set_current_projectid(request.ctx.userid, request.args["projectid"][0])
                data = pipe_producer.current_pipe_state(request.ctx.userid, request.args["projectid"][0])
                if data is None:
                    info = f"Something unexpected happened while setting the current project to {request.args['projectid'][0]}"
                    status = 500
                else:
                    # print(data)
                    # print(request.args["projectid"][0])
                    data["id"] = request.args["projectid"][0]
                    data["name"] = proj[0]
                    data["timestamp"] = proj[1]
                    info = f"Successfully set {proj[0]} as the current project"
                    status = 200

    except Exception as ex:
        info = ex.args[0]
        status = 500
        # import traceback

        # print(traceback.format_exc())
    finally:
        # print(data)
        return response.json(
            {
                "success": True if (status == 200) else False,
                "version": "v1",
                "info": info,
                "data": data if (status == 200) else {},
            },
            status=status,
        )
