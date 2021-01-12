from .app import app
from .config import server_config

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    app.config.KEEP_ALIVE_TIMEOUT = server_config.app.KEEP_ALIVE_TIMEOUT
    app.config.REQUEST_MAX_SIZE = server_config.app.REQUEST_MAX_SIZE
    app.config.RESPONSE_TIMEOUT = server_config.app.RESPONSE_TIMEOUT
    app.config.REQUEST_BUFFER_QUEUE_SIZE = server_config.app.REQUEST_BUFFER_QUEUE_SIZE
    app.run(
        host=server_config.app.host,
        port=server_config.app.port,
        workers=server_config.app.workers,
        auto_reload=server_config.app.auto_reload,
        debug=server_config.app.debug,
        access_log=server_config.app.access_log,
    )
