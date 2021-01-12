import redis

from ..store.backend.redis import Application, ModelsTasksProducer, PipeTasksProducer
from .config import server_config

# get redis host, port, and db from Conf object
redis_pool = redis.ConnectionPool(**server_config.redis.dict())

pipe_producer = PipeTasksProducer(redis_pool)

model_producer = ModelsTasksProducer(redis_pool)

store_backend = Application(redis_pool)
