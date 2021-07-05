from multiprocessing import Process

from ..store.backend.redis import PipeTasksConsumer
from .env import env
from .stages import build, consume, prepare, transform

# funcion to process pipe stages
stage_func_table = {
    "consume:POST": consume.process,
    "prepare:POST": prepare.process,
    "transform:POST": transform.process,
    "build:POST": build.process,
}

ps = []


# spawn this worker func onto a process
def pipe_consumer_func(worker_name):
    import warnings

    warnings.simplefilter("ignore")

    consumer = PipeTasksConsumer(env().redis_config, stage_func_table)
    try:
        consumer.run(worker_name)
    except Exception as ex:
        if not (isinstance(ex, KeyboardInterrupt)):
            print("-------------Exception happened in pipe worker: ", worker_name)
            print(ex)


def spawn_pipe_workers():
    for worker_name in env().workers:
        p = Process(target=pipe_consumer_func, args=(worker_name,))
        p.start()
        ps.append(p)


def join_pipe_workers():
    for p in ps:
        p.join()


# SIGKILL process (no cleanup)
def kill_pipe_workers():
    for i, p in enumerate(ps):
        p.kill()
        p.join()
    print("============ KILLED PIPELINE WORKERS (MLPIPELINE) ============")
