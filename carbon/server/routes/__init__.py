from sanic import Blueprint
from .auth import auth_bp
from .projects import projects_bp
from .uploads import uploads_bp
from .sse import sse_bp
from .pipe import pipe_bp
from .models import models_bp
from .checks import checks_bp

root_bp = Blueprint.group(
    checks_bp,
    auth_bp,
    projects_bp,
    uploads_bp,
    pipe_bp,
    models_bp,
    sse_bp,
    url_prefix="/",
)
