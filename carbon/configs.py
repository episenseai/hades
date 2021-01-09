from pathlib import Path
from typing import List

from pydantic import BaseModel  # ValidationError
# from pydantic.types import UUID4  # FilePath


class MLModel(BaseModel):
    modelid: str  #UUID4
    modelname: str
    filename: str  # FilePath


class RegressorModels(BaseModel):
    models: List[MLModel]


class ClassifierModels(BaseModel):
    models: List[MLModel]


regressors_path = Path('configs/models/regressor_models.json')
regressors = RegressorModels.parse_file(regressors_path)

classifiers_path = Path('configs/models/classifier_models.json')
classifiers = ClassifierModels.parse_file(classifiers_path)

# debug(regressors, classifiers)


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
    temp_folder: str
    num_pipe_workers: int
    pipe_worker_names: List[str]
    num_model_workers: int
    model_worker_names: List[str]


jobsconfig_path = Path('configs/jobs.json')
jobsconfig = JobsConfig.parse_file(jobsconfig_path)
# debug(jobsconfig)


class ServerConfig(BaseModel):
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
    CORS: List[str]
    CORSmethods: List[str]


serverconfig_path = Path('configs/server.json')
serverconfig = ServerConfig.parse_file(serverconfig_path)
# debug(serverconfig)


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    decode_responses: bool


redisconfig_path = Path('configs/redis.json')
redisconfig = RedisConfig.parse_file(redisconfig_path)
# debug(redisconfig)
