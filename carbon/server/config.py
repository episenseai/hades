from pathlib import Path
from typing import List

from pydantic import BaseModel

config_path = Path('configs/server.json')


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    decode_responses: bool


class AppConfig(BaseModel):
    host: str
    port: int
    workers: int
    auto_reload: bool
    debug: bool
    access_log: bool
    KEEP_ALIVE_TIMEOUT: int
    REQUEST_MAX_SIZE: int
    RESPONSE_TIMEOUT: int
    REQUEST_TIMEOUT: int
    REQUEST_BUFFER_QUEUE_SIZE: int
    env: str
    data: str
    CORS_origins: List[str]
    CORS_methods: List[str]


class JobsConfig(BaseModel):
    models_folder: str
    uploads_folder: str


class ServerConfig(BaseModel):
    redis: RedisConfig
    app: AppConfig
    jobs: JobsConfig


server_config = ServerConfig.parse_file(config_path)
# debug(server_config)
