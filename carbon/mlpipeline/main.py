from .app import join_pipe_workers, kill_pipe_workers, spawn_pipe_workers
from .utils import printBox

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    try:
        spawn_pipe_workers()
        printBox("Running [[ carbon :: ML PIPELINE ]] app")
        join_pipe_workers()
    except KeyboardInterrupt:
        print("\n")
        printBox("Terminated ML PIPELINE workers....................................")
    finally:
        kill_pipe_workers()
