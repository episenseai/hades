from .app import spawn_model_workers, join_model_workers, kill_model_workers

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    try:
        spawn_model_workers()
        join_model_workers()
    except KeyboardInterrupt:
        print("\nShutting Down model workers")
        pass
    finally:
        kill_model_workers()
