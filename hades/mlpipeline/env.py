from functools import lru_cache

from pydantic import BaseSettings, SecretStr, root_validator
from pydantic.types import PositiveInt


class Settings(BaseSettings):
    # DEV or PRODUCTION
    ENV: str = "DEV"

    REDIS_PASSWORD: SecretStr = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DATABASE_NUMBER: int = 3

    UPLOADS_VOLUME: str = "./bucket/uploads"
    TMP_VOLUME: str = "./bucket/tmp"

    NUM_WORKERS: PositiveInt = 1
    # export HADES_MLPIPELINE_WORKER_NAMES='["worker1", "worker2"]'
    WORKER_NAMES: list[str] = [
        "pipeline:worker1",
        "pipeline:worker2",
        "pipeline:worker3",
        "pipeline:worker4",
    ]

    @root_validator(pre=False, skip_on_failure=True)
    def duplicate_name(cls, values):
        items = []
        for item in values.get("WORKER_NAMES"):
            if item in items:
                raise ValueError(f"Duplicate worker name => {item}")
            else:
                items.append(item)
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def workers_len(cls, values):
        num_workers = values.get("NUM_WORKERS")
        worker_names = values.get("WORKER_NAMES")
        if num_workers > len(worker_names):
            raise ValueError(
                f"Number of NUM_WORKERS={num_workers} > length of WORKER_NAMES={len(worker_names)}"
            )
        return values

    @property
    def workers(self) -> list[str]:
        return self.WORKER_NAMES[: min(self.NUM_WORKERS, len(self.WORKER_NAMES))]

    @property
    def redis_url(self) -> str:
        # redis://localhost:6379/3
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DATABASE_NUMBER}"

    @property
    def redis_config(self) -> dict:
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DATABASE_NUMBER,
            "password": self.REDIS_PASSWORD.get_secret_value(),
            "decode_responses": True,
        }

    class Config:
        env_prefix = "HADES_MLPIPELINE_"
        case_sensitive = True


@lru_cache
def env() -> Settings:
    return Settings()
