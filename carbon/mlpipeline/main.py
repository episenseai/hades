from .app import spawn_pipe_workers, join_pipe_workers, kill_pipe_workers

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    try:
        spawn_pipe_workers()
        join_pipe_workers()
    except KeyboardInterrupt:
        print("\nShutting Down pipe workers")
        pass
    finally:
        kill_pipe_workers()
