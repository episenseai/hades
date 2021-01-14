from ..store.backend.redis import Application, ModelsTasksProducer, PipeTasksProducer
from .config import server_config

# get redis host, port, and db from Conf object

pipe_producer = PipeTasksProducer(server_config.redis.dict())

model_producer = ModelsTasksProducer(server_config.redis.dict())

store_backend = Application(server_config.redis.dict())
