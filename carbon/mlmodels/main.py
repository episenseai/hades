from .app import join_model_workers, kill_model_workers, spawn_model_workers

if __name__ == "__main__":
    try:
        spawn_model_workers()
        print("============ STARTING (MLMODELS) ============")
        join_model_workers()
    except KeyboardInterrupt:
        pass
    finally:
        kill_model_workers()
