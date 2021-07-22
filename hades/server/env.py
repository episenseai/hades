from functools import lru_cache

from pydantic import BaseSettings, SecretStr, validator


class Settings(BaseSettings):
    # DEV or PRODUCTION
    ENV: str = "DEV"

    PORT: int = 3002
    WORKERS: int = 1

    CORS_ORIGIN: str = "http://localhost:3000"

    REDIS_PASSWORD: SecretStr = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DATABASE_NUMBER: int = 3

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
    def redis_config(self) -> dict:
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DATABASE_NUMBER,
            "password": self.REDIS_PASSWORD.get_secret_value(),
            "decode_responses": True,
        }

    @property
    def cors_origins(self) -> list[str]:
        return [self.CORS_ORIGIN]

    @property
    def cors_methods(self) -> list[str]:
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    class Config:
        env_prefix = "HADES_SERVER_"
        case_sensitive = True


@lru_cache
def env() -> Settings:
    return Settings()
