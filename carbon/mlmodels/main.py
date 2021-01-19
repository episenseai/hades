from .app import join_model_workers, kill_model_workers, spawn_model_workers
from .utils import printBox

if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")

    try:
        spawn_model_workers()
        printBox("Running [[ carbon :: ML MODELS ]] app")
        join_model_workers()
    except KeyboardInterrupt:
        print("\n")
        printBox("Terminated ML MODELS workers....................................")
    finally:
        kill_model_workers()
