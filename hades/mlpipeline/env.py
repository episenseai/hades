from enum import Enum, unique
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, SecretStr, root_validator, validator
from pydantic.types import PositiveInt


@unique
class Env(str, Enum):
    DEV = "DEV"
    PRODUCTION = "PRODUCTION"


class Settings(BaseSettings):
    ENV: Env = Env.DEV

    REDIS_PASSWORD: Optional[SecretStr] = None
    REDIS_HOST: str = "localhost"
    # User setting is ignored and it always uses the deault value
    REDIS_PORT: int = 6379
    REDIS_DATABASE_NUMBER: int = 2

    UPLOADS_VOLUME: str = "./bucket/uploads"
    TMP_VOLUME: str = "./bucket/mlpipeline/tmp"

    NUM_WORKERS: PositiveInt = 1
    # export HADES_MLPIPELINE_WORKER_NAMES='["worker1", "worker2"]'
    WORKER_NAMES: list[str] = [
        "mlpipeline:worker1",
        "mlpipeline:worker2",
        "mlpipeline:worker3",
        "mlpipeline:worker4",
    ]

    @validator("REDIS_PORT", pre=True, always=True)
    def ignore_redis_port(cls, _):
        """Always run on default port. Ignore environement"""
        return 6379

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
        # TODO: This might cause problems when running multiple copies of this app
        # because of the overlap in worker names; every copy of this application
        # will be using the same worker name to connect with the redis streams.
        return self.WORKER_NAMES[: min(self.NUM_WORKERS, len(self.WORKER_NAMES))]

    @property
    def redis_url(self) -> str:
        # redis://localhost:6379/3
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DATABASE_NUMBER}"

    @property
    def redis_pasword(self) -> Optional[str]:
        secret = self.REDIS_PASSWORD
        if secret is None:
            return None
        else:
            return secret.get_secret_value()

    @property
    def redis_config(self) -> dict:
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DATABASE_NUMBER,
            "password": self.redis_pasword,
            "decode_responses": True,
        }

    class Config:
        env_prefix = "HADES_MLPIPELINE_"
        case_sensitive = True


@lru_cache
def env() -> Settings:
    return Settings()
