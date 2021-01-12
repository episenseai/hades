from ..store.backend.redis import Application, PipeTasksProducer, ModelsTasksProducer
from .config import server_config
import redis

# get redis host, port, and db from Conf object
redis_pool = redis.ConnectionPool(**server_config.redis.dict())

pipe_producer = PipeTasksProducer(redis_pool)

model_producer = ModelsTasksProducer(redis_pool)

store_backend = Application(redis_pool)
