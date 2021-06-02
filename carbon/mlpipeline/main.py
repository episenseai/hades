from .app import join_pipe_workers, kill_pipe_workers, spawn_pipe_workers

if __name__ == "__main__":
    try:
        spawn_pipe_workers()
        print("============ STARTING (MLPIPELINE) ============")
        join_pipe_workers()
    except KeyboardInterrupt:
        pass
    finally:
        kill_pipe_workers()
