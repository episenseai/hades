from pathlib import Path
from typing import List

from pydantic import BaseModel, root_validator
from pydantic.types import PositiveInt

config_path = Path('configs/mlmodels.json')


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    decode_responses: bool


class WorkersConfig(BaseModel):
    num_workers: PositiveInt
    worker_names: List[str]

    @root_validator(pre=False, skip_on_failure=True)
    def duplicate_name(cls, values):
        items = []
        for item in values.get('worker_names'):
            if item in items:
                raise ValueError(f"Duplicate worker name => {item} in '{config_path}'")
            else:
                items.append(item)
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def workers_len(cls, values):
        num_workers = values.get('num_workers')
        worker_names = values.get('worker_names')
        if num_workers > len(worker_names):
            raise ValueError(
                f"Number of num_workers={num_workers} > total worker_names={len(worker_names)} in '{config_path}'"
            )
        return values


class JobsConfig(BaseModel):
    uploads_folder: str


class MLModelsConfig(BaseModel):
    redis: RedisConfig
    workers: WorkersConfig
    jobs: JobsConfig


mlmodels_config = MLModelsConfig.parse_file(config_path)
# debug(mlmodels_config["redis"])
