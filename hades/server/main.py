from .app import app
from .env import env

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    app.config.KEEP_ALIVE_TIMEOUT = 30
    app.config.REQUEST_MAX_SIZE = 100000000
    app.config.RESPONSE_TIMEOUT = 300
    app.config.REQUEST_BUFFER_QUEUE_SIZE = 400

    if env().ENV == "DEV":
        AUTO_RELOAD = True
    else:
        AUTO_RELOAD = False

    app.run(
        host="0.0.0.0",
        port=env().PORT,
        workers=1,
        auto_reload=AUTO_RELOAD,
        debug=False,
        access_log=True,
    )
