from ..store.backend.redis import Application, ModelsTasksProducer, PipeTasksProducer
from .env import env

# get redis host, port, and db from Conf object

pipe_producer = PipeTasksProducer(env().redis_config)

model_producer = ModelsTasksProducer(env().redis_config)

store_backend = Application(env().redis_config)
