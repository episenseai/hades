from pathlib import Path
from typing import List

from pydantic import BaseModel

regressors_path = Path('configs/store/regressor_models.json')
classifiers_path = Path('configs/store/classifier_models.json')
multi_classifiers_path = Path('configs/store/multi_classifier_models.json')


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
# debug(regressors)
# debug(classifiers)
# debug(multi_classifiers)

jobqueue_path = Path("configs/store/queue.json")


class JobQueueConfig(BaseModel):
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


jobqueue_config = JobQueueConfig.parse_file(jobqueue_path)
# debug(jobqueue_config)
