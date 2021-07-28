from enum import Enum, unique
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, SecretStr, validator


@unique
class Env(str, Enum):
    DEV = "DEV"
    PRODUCTION = "PRODUCTION"


class Settings(BaseSettings):
    ENV: Env = Env.DEV

    # User setting is ignored and it always uses the deault value
    PORT: int = 3002
    WORKERS: int = 1

    CORS_ENABLED: bool = True
    CORS_ORIGIN: str = "http://localhost:3000"

    REDIS_PASSWORD: Optional[SecretStr] = None
    REDIS_HOST: str = "localhost"
    # User setting is ignored and it always uses the deault value
    REDIS_PORT: int = 6379
    REDIS_DATABASE_NUMBER: int = 2

    UPLOADS_VOLUME: str = "./bucket/uploads"
    MODELS_VOLUME: str = "./bucket/models"

    @validator("PORT", pre=True, always=True)
    def ignore_port(cls, _):
        """Always run on default port. Ignore environement"""
        return 3002

    @validator("REDIS_PORT", pre=True, always=True)
    def ignore_redis_port(cls, _):
        """Always run on default port. Ignore environement"""
        return 6379

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
        env_prefix = "HADES_SERVER_"
        case_sensitive = True


@lru_cache
def env() -> Settings:
    return Settings()
