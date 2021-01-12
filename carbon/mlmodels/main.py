from .app import join_model_workers, kill_model_workers, spawn_model_workers

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")

    try:
        spawn_model_workers()
        join_model_workers()
    except KeyboardInterrupt:
        print("\nShutting Down model workers")
        pass
    finally:
        kill_model_workers()
