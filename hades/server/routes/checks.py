from sanic import Blueprint, response

checks_bp = Blueprint("checks_service", url_prefix="/checks")


# route: /checks/health
@checks_bp.get("/health")
async def health(_request):
    return response.json(
        {
            "success": 200,
            "version": "v1",
            "info": "Health checks...",
            "data": {},
        },
        status=200,
    )
