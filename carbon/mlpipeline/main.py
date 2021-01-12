from multiprocessing import Process

import redis

from carbon.redis_task import PipeTasksConsumer

from .config import mlpipeline_config
from .stages import build, consume, prepare, transform

# funcion to process pipe stages
func_dict = {
    "consume:POST": consume.process,
    "prepare:POST": prepare.process,
    "transform:POST": transform.process,
    "build:POST": build.process,
}


# spawn this worker func onto a process
def pipe_consumer_func(worker):
    redis_pool = redis.ConnectionPool(**mlpipeline_config.redis.dict())
    consumer = PipeTasksConsumer(redis_pool, func_dict)
    try:
        consumer.run(worker)
    except Exception as ex:
        if not (isinstance(ex, KeyboardInterrupt)):
            print("-------------Exception happened in pipe worker: ", worker)
            print(ex)


# SIGKILL process (no cleanup)
def kill_pipe_workers():
    for i, p in enumerate(ps):
        p.kill()
        p.join()
        print(
            f"─────────shutdown pipe worker = {allowed_pipe_workers[i]} pid = {p.pid} exitcode = {p.exitcode} is_alive = {p.is_alive()}"
        )


# list of workers
allowed_pipe_workers = mlpipeline_config.workers.worker_names[:(
    min(mlpipeline_config.workers.num_workers, len(mlpipeline_config.workers.worker_names)))]
ps = []


def spawn_pipe_workers():
    for worker in allowed_pipe_workers:
        p = Process(target=pipe_consumer_func, args=(worker,))
        p.start()
        ps.append(p)

    return kill_pipe_workers
