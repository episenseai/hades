from .app import join_model_workers, kill_model_workers, spawn_model_workers
from .env import env

if __name__ == "__main__":
    try:
        spawn_model_workers()
        print("============ STARTING (MLMODELS) ============")
        print(f"{env()}")
        join_model_workers()
    except KeyboardInterrupt:
        pass
    finally:
        kill_model_workers()
