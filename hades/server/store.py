from ..store.backend.redis import Application, ModelsTasksProducer, PipeTasksProducer
from ..store.metrics import MetricsDB
from .env import Env, env
from .logger import logger

pipe_producer = PipeTasksProducer(env().redis_config)

model_producer = ModelsTasksProducer(env().redis_config)

store_backend = Application(env().redis_config)

# XXX:
if env().ENV == Env.DEV:
    MODELS_PER_NUM_HOUR = 50
else:
    MODELS_PER_NUM_HOUR = env().MODELS_PER_NUM_HOUR

metrics_db = MetricsDB(
    redis_host=env().REDIS_METRICS_HOST,
    redis_port=env().REDIS_METRICS_PORT,
    redis_password=env().redis_metrics_password,
    redis_db=env().REDIS_METRICS_DATABASE_NUMBER,
    num_hour=env().NUM_HOUR,
    models_per_num_hour=MODELS_PER_NUM_HOUR,
    logger=logger,
)
