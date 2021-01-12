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
    USERS_HASHMAP: str
    USERS_SET: str
    USERS_ID: str
    PROJECTS_SORTED_SET: str
    CURRENT_PROJECT: str
    UPLOADS_SORTED_SET: str
    PROJECTS_DESC_HASHMAP: str
    MODELS_SORTED_SET: str
    DB_GEN: int
    TEST_USER_ID: str
    TEST_PROJ_ID: str
    pipe_queue: str
    models_queue: str
    jwt: str
    models_folder: str
    uploads_folder: str


class ServerConfig(BaseModel):
    redis: RedisConfig
    app: AppConfig
    jobs: JobsConfig


server_config = ServerConfig.parse_file(config_path)
# debug(server_config)

regressors_path = Path('configs/models/regressor_models.json')
classifiers_path = Path('configs/models/classifier_models.json')
multi_classifiers_path = Path('configs/models/multi_classifier_models.json')


class MLModel(BaseModel):
    modelid: str  #UUID4
    modelname: str
    filename: str  # FilePath


class RegressorModels(BaseModel):
    models: List[MLModel]


regressors = RegressorModels.parse_file(regressors_path)


class ClassifierModels(BaseModel):
    models: List[MLModel]


classifiers = ClassifierModels.parse_file(classifiers_path)


class MultiClassifierModels(BaseModel):
    models: List[MLModel]


multi_classifiers = MultiClassifierModels.parse_file(classifiers_path)
