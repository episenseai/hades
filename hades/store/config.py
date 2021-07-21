import pathlib
from typing import List, Optional

from pydantic import BaseModel

file_dir = pathlib.Path(__file__).parent.resolve()


class MLModel(BaseModel):
    modelid: str  # UUID4
    modelname: str
    filename: str  # FilePath
    modelType: Optional[str] = None


class RegressorModels(BaseModel):
    models: List[MLModel]


regressors = RegressorModels.parse_file(file_dir / "regressor_models.json")


class ClassifierModels(BaseModel):
    models: List[MLModel]


classifiers = ClassifierModels.parse_file(file_dir / "classifier_models.json")


class MultiClassifierModels(BaseModel):
    models: List[MLModel]


multi_classifiers = MultiClassifierModels.parse_file(file_dir / "multi_classifier_models.json")


class JobQueueConfig(BaseModel):
    PROJECTS_SORTED_SET: str
    CURRENT_PROJECT: str
    UPLOADS_SORTED_SET: str
    PROJECTS_DESC_HASHMAP: str
    MODELS_SORTED_SET: str
    DB_GEN: int
    PIPE_QUEUE: str
    MODELS_QUEUE: str


jobq_setting = JobQueueConfig.parse_file(file_dir / "queue.json")
