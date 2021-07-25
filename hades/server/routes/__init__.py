from sanic import Blueprint

from .checks import checks_bp
from .models import models_bp
from .pipe import pipe_bp
from .projects import projects_bp
from .sse import sse_bp
from .uploads import uploads_bp

root_bp = Blueprint.group(
    projects_bp,
    uploads_bp,
    pipe_bp,
    models_bp,
    sse_bp,
    url_prefix="/",
)
