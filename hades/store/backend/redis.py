import base64
import datetime
import gzip
import hashlib
import pickle
import time
import traceback
import uuid
from collections import Counter, namedtuple
from multiprocessing import Process, Queue
from typing import Any, List, Optional, Tuple

import orjson
import redis
from pydantic import BaseModel

from ..config import MLModel, classifiers, jobq_setting, multi_classifiers, regressors

seq = [
    "consume:GET",
    "consume:POST",
    "prepare:GET",
    "prepare:POST",
    "transform:GET",
    "transform:POST",
    "build:GET",
    "build:POST",
    "finalconfig:GET",
]

for model in regressors.models:
    model.modelType = "regressor"
for model in classifiers.models:
    model.modelType = "classifier"
for model in multi_classifiers.models:
    model.modelType = "multi_classifier"


class TaskResult:
    def __init__(self, result: Optional[Any] = None, exception=None):
        self.result = result
        self.exception = exception


class TaskCancelled:
    def __init__(self, status: bool = False, exception: Optional[Tuple[Exception, str]] = None):
        self.status = status
        self.exception = exception


def main_task_wrapper(func, arg, result_queue: Queue):
    import warnings

    warnings.simplefilter("ignore")

    try:
        result = func(arg)
        # debug(result)
        # import time
        # from random import randint

        # slow down the task for debugging
        # time.sleep(randint(20, 30))
        result_queue.put(TaskResult(result=result), block=True, timeout=120)
    except Exception as ex:
        tb = traceback.format_exc()
        result_queue.put(TaskResult(exception=(ex, tb)), block=True, timeout=120)


def status_task_wrapper(result_queue: Queue, redis_config, cancelled_hashmap, jobid):
    try:
        pool = redis.Redis(connection_pool=redis.ConnectionPool(**redis_config))
        while True:
            cancelled_status = int(pool.hget(cancelled_hashmap, jobid))  # type: ignore
            if cancelled_status != 0:
                status = True
                break
            time.sleep(2)
        result_queue.put(TaskCancelled(status=status), block=True, timeout=120)
    except Exception as ex:
        tb = traceback.format_exc()
        result_queue.put(TaskCancelled(exception=(ex, tb)), block=True, timeout=120)


def execute_task(func, arg, redis_config, cancelled_hashmap, jobid):
    result_queue = Queue()
    main_task = Process(
        target=main_task_wrapper,
        args=(
            func,
            arg,
            result_queue,
        ),
    )
    main_task.start()
    status_task = Process(
        target=status_task_wrapper,
        args=(
            result_queue,
            redis_config,
            cancelled_hashmap,
            jobid,
        ),
    )
    status_task.start()
    result = result_queue.get()

    try:
        for p in [main_task, status_task]:
            p.kill()
            p.join()
    except Exception as ex:
        print("Exception happened during killing: ", ex)

    if isinstance(result, TaskCancelled):
        if result.exception:
            error_msg = f"CANCELLATION ERROR:\n{result.exception[0]}\n{result.exception[1]}"
            print(error_msg, jobid)
            # (result: Option[Any], error_msg: str, cancelled: bool, exception: bool)
            return (None, error_msg, result.status, True)
        else:
            error_msg = "TASK CANCELLED"
            print(error_msg, jobid)
            return (None, error_msg, result.status, False)
    else:
        if result.exception:
            error_msg = f"TASK ERROR:\n{result.exception[0]}\n{result.exception[1]}"
            print(error_msg, jobid)
            return (result.result, error_msg, False, True)
        else:
            error_msg = ""
            return (result.result, error_msg, False, False)


class RedisTasks:
    def __init__(self, redis_config: dict, queue):
        """
        pool: redis ConnectionPool
        queue: one of:
            "pipe" -> for pipe jobs
            "models" -> individual model build jobs
        """

        self.redis_config = redis_config
        self.redis = redis.Redis(**redis_config)
        self.jobq = f"{queue}:{jobq_setting.DB_GEN}"
        self.jobq_CG = f"CG:{self.jobq}"
        self.deadq = f"dead.{queue}:{jobq_setting.DB_GEN}"
        self.deadq_CG = f"CG:{self.deadq}"

        # check if we are able to talk to redis
        assert self.redis.ping(), "Can not Ping redis server"

        for q, cg in [(self.jobq, self.jobq_CG), (self.deadq, self.jobq_CG)]:
            if not self.redis.exists(q) == 1:
                # create a jobq
                self.redis.xadd(q, {"OK": "OK"})
            # check if jobq is a stream
            assert self.redis.type(q) == "stream", f"jobq_queue = {q} nust be a stream"
            if not any((q_cg["name"] == cg for q_cg in self.redis.xinfo_groups(self.jobq))):
                # create consumer group
                self.redis.xgroup_create(q, cg)

    @staticmethod
    def to_JSON(item):
        """
        to convert model results dictionary to json
        """
        try:
            data = orjson.dumps(item, option=orjson.OPT_SERIALIZE_NUMPY)
            # debug(data)
            return data
        except Exception as ex:
            raise JSONEncodeError from ex

    @staticmethod
    def from_JSON(item):
        """
        to parse JSON config data from redis into python dictionary
        """
        try:
            data = orjson.loads(item)
            # debug(data)
            return data
        except Exception as ex:
            raise JSONDecodeError from ex

    @staticmethod
    def pickle_gzip(obj):
        """
        Pickle dump the object and base64 encode it to store on the redis
            - helpful in storing trained models produced during production
        """
        try:
            return gzip.compress(pickle.dumps(obj, protocol=4))
        except Exception as ex:
            raise PickleError from ex

    @staticmethod
    def unpickle_unzip(obj_dump_encoded):
        """
        base64 decode the bytes got from redis and then unpickle into python object
            - helpful in reloading trained models for prediction
        """
        try:
            return pickle.loads(gzip.decompress(obj_dump_encoded))
        except Exception as ex:
            raise UnpickleError from ex

    @staticmethod
    def encode(obj):
        return gzip.compress(base64.b64encode(gzip.compress(pickle.dumps(obj, protocol=4))))

    @staticmethod
    def decode(bbytes):
        return pickle.loads(gzip.decompress(base64.b64decode(gzip.decompress(bbytes))))

    def hashmap(self, user_id, project_id):
        return f"{user_id}:{project_id}:{self.jobq}"


class RedisTasksError(Exception):
    def __init__(self, msg="RedisTasksError"):
        super().__init__(msg)
        self.stack_trace = traceback.format_exc()


class JSONEncodeError(RedisTasksError):
    pass


class JSONDecodeError(RedisTasksError):
    pass


class PickleError(RedisTasksError):
    pass


class UnpickleError(RedisTasksError):
    pass


class PipeToModelsError(RedisTasksError):
    pass


class ConsumerError(RedisTasksError):
    pass


### NAMING CONVENTIONS
#
# pipe queue -> pipe:1572618712, consumer group -> CG:pipe:1572618712
#      dead -> deadpipe:1572618712, consumer group -> CG:deadpipe:1572618712
# pipe queue fields
#       stage: one of [consume:POST, ..., build:POST]
#       userid
#       projectid
#
# models queue -> models:1572618712, consumer group -> CG:models:1572618712
#        dead -> deadmodels:1572618712, consumer group -> CG:deadmodels:1572618712
# models queue fields
#        modelid -> (unique id for each model)
#        modelname -> (Cannonical name of the model)
#        config ()
#
#
# Results (Redis Hash)
# pipe results ->  userid:projectid:pipe:1572618712
# pipe results fields:
#       consume:GET -> data (stringified JSON)
#       consume:POST -> data
#       prepare:GET -> data
#       prepare:POST -> data
#       transform:GET -> data
#       transform:POST -> data
#       build:GET -> data
#       build:POST -> data
#       finalconfig:GET -> data (final config data)
#       consume:POST:jobid -> id (returned from stream insertion)
#       prepare:POST:jobid -> id
#       transform:POST:jobid -> id
#       build:POST:jobid -> data
#       current.stage -> one of [consume:GET, consume:POST, ..., build:GET, build:POST, finalconfig:GET]
#               - in last stage, build:POST, job will be to insert jobs for each model in
#                 models:1572618712 queue, and
#               - set current:stage = finalconfig:GET to signal that pipe is complete and move on to models
#       pipe:STATUS: 1 or 0
#               -1  -> error in the pipeline
#                0  -> everything is ok (set initially to start the pipeline)
#                1  -> complete (finalconfig:GET job done; data added to Redis)
#       error:TYPE ->
#       error:STACK ->
#
#   steps:
#       1. each stage:POST:jobid will be added to the the stream and set current:stage of results
#       2: worker wil pull the job - (jobid + queue_data) and perform the job
#       3: check current:stage == queue_data.stage && stage:POST:jobid == jobid
#               (make sure that previous job was not cancelled or user navigates to the previous stage)
#       4: if true add results to -> next(stage):GET:jobid and set current:stage -> next(stage):GET
#       5: contnuously poll for the current:stage to serve the GET request on the next stage
# data specification for the fields of the pipe stages
#   consume:GET
#       empty (no data)
#   consume:POST
#       steps:
#           1. (on the interface do a file upload, or select a previous file upload)
#           2. then do a POST request to the server with filelocation/name to store consumer:POST data
#       data:
#           filename (returned after successfull upload)
#   prepare:GET
#       data: same as mock_data on the frontend
#   prepare:POST
#       steps:
#           1. do a POST request after the making changes on the frontend
#       data:
#           for each column
#               column_id
#               column_name
#               data_type
#               imputable
#           target_column
#   transform:GET
#       data: same format as mock_data on the frontend
#   transform:POST
#       data:
#           for each column
#               column_id
#               column_name
#               include
#               weight
#   build:GET
#       data: same format ad mock_data
#   build:POST
#       data:
#           sampling
#           cv_folds
#           holdout
#           downsammple
#           metric
#
# After build:
#
# Mapping model builds to userid and projectid - (Redis Hash)
#  - to determine the location where to put the model results
#  - after model job is done, worker posts the result containing:
#        id -> id of the job pulled out of the queue
#        modelid -> from the model id in the job description
#        results -> actual results of the processing
#
# After server recieves the result:
#   1. it checks against the map:buildtask:project to get the key
#   2. if the key exists pull off the userid and projectid and save the result to mongodb
#           include the jobid timestamp while saving to mongodb
#   3. XACK the models job queue for the completion of the job
#
# map:buildtask:project:1572618712
#   id1:modelid-x -> userid-1:projectid-1:models:1572618712 (id1 -> job id returned adding model to queue
#                          modelid-x -> unique model identifier)
#   id2:modelid-y -> userid-2:projectid-2:models:1572618712
#   ... -> ...
#
#
# When submitting data to the backend run some primitive checks on the data
#   - see if the JSON is parsable

# handle calls from frontend
# used to submut jobs


class RedisTasksProducer(RedisTasks):
    pass


class RedisTasksConsumer(RedisTasks):
    def __init__(self, redis_config, queue):
        super().__init__(redis_config, queue)
        self._item = None
        self.pending_jobs = True

    @property
    def jobid(self):
        return self._item[0]

    # read an item of the queue (blocking)
    # '0' => process the pending message '>' => process the new messages
    def xreadgroup(self, consumer_name, nextjob=">"):
        self._item = None
        # print("getting job for ", consumer_name, self.jobq, self.jobq_CG)
        return self.redis.xreadgroup(
            self.jobq_CG,
            consumer_name,
            {self.jobq: nextjob},
            count=1,
            block=0,
        )[0][1]

    def get_item(self, consumer_name):
        items = []
        if self.pending_jobs:
            items = self.xreadgroup(consumer_name, nextjob="0")
            if items:
                print("pending items")
        if not items:
            self.pending_jobs = False
            items = self.xreadgroup(consumer_name)
            print("new items")
        # print(items)
        self._item = items[0]

    def reset_item(self):
        self._item = None


class PipeTasksProducer(RedisTasksProducer):
    def __init__(self, redis_config):
        # print("instantiating pipe producer ....")
        super().__init__(redis_config, "pipe")
        lua_job_submit = """
            local jobid =  redis.call('xadd', KEYS[1], '*', 'stage', KEYS[2], 'result_hashmap', KEYS[3])
            local res1 = redis.call('hset', KEYS[3], KEYS[2], ARGV[1])
            local res2 = redis.call('hset', KEYS[3], KEYS[4], jobid)
            local res3 = redis.call('hset', KEYS[3], 'current.stage', KEYS[2])
            local res4 = redis.call('hsetnx', KEYS[3], 'pipe:STATUS', 0)
            return {jobid, res1, res2, res3, res4}
        """
        lua_job_submit = lua_job_submit.strip()
        self.job_sha = self.redis.script_load(lua_job_submit)

    def submit_job(self, user_id, project_id, stage, data_dict):
        """
        Submit a POST job for the pipeline
            - add job descripption to the queue
            - add job data to the project hash set
        Params:
            - user_id: supplied from the frontend
            - project_id: supplied from the frontend
            - stage: one of the pipeline stages
            - data_dict: data which will be converted to JSON and stored on redis
        """
        result_hashmap = self.hashmap(user_id, project_id)
        cuurent_jobid_key = f"{stage}:jobid"

        result = self.redis.evalsha(
            self.job_sha,
            4,
            self.jobq,
            stage,
            result_hashmap,
            cuurent_jobid_key,
            self.to_JSON(data_dict),
        )
        return result

    def unfreeze_pipe(self, user_id, project_id):
        status = True
        result_hashmap = self.hashmap(user_id, project_id)
        with self.redis.pipeline() as pipe:
            error_count = 0
            while True:
                try:
                    pipe.watch(result_hashmap)
                    pstatus = pipe.hget(result_hashmap, "pipe:STATUS")
                    current_stage = self.redis.hget(result_hashmap, "current.stage")
                    idx = seq.index(current_stage) - 1  # type: ignore
                    # print(status, current_stage, idx)
                    if idx < 0:
                        status = False
                        print(status, current_stage, idx)
                        break
                    # after WATCHing, the pipeline is put into i
                    if (int(pstatus)) == -1:
                        pipe.multi()
                        pipe.hset(result_hashmap, "pipe:STATUS", 0)
                        pipe.hset(result_hashmap, "current.stage", seq[idx])
                        pipe.execute()
                    break
                except redis.WatchError:
                    if error_count > 5:
                        status = False
                        break
                    error_count += 1
                    continue
        return status

    def setup_new_pipe(self, userid, projectid):
        result_hashmap = self.hashmap(userid, projectid)
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    pipe.watch(result_hashmap)
                    pipe.multi()
                    pipe.hsetnx(
                        result_hashmap,
                        "consume:GET",
                        self.to_JSON({"stage": "consume:GET", "data": {}}),
                    )
                    pipe.hsetnx(result_hashmap, "current.stage", "consume:GET")
                    pipe.hsetnx(result_hashmap, "pipe:STATUS", 0)
                    pipe.hsetnx(result_hashmap, "error:TYPE", "")
                    pipe.hsetnx(result_hashmap, "error:STACK", "")
                    pipe.execute()
                    # generate a different project_id
                    break
                except redis.WatchError as ex:
                    if watch_error_count > 100:
                        print(ex)
                        raise Exception(
                            "Something fatal happened while creating a new project"
                        ) from ex
                    if watch_error_count > 20:
                        time.sleep(0.1)
                    watch_error_count += 1
                    continue

    def current_pipe_state(self, userid, projectid, include_error=False):
        result_hashmap = self.hashmap(userid, projectid)
        result = None
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    pipe.watch(result_hashmap)
                    pipe.multi()
                    pipe.hmget(
                        result_hashmap,
                        "current.stage",
                        "pipe:STATUS",
                        "error:TYPE",
                        "error:STACK",
                    )
                    result, *_ = pipe.execute()
                    if any(map(lambda x: x is None, result)):
                        break
                    if include_error:
                        result = {
                            "current_stage": result[0],
                            "pipe_status": result[1],
                            "error_stype": result[2],
                            "error_stack": result[3],
                        }
                    else:
                        result = {
                            "current_stage": result[0],
                            "pipe_status": result[1],
                            "error_stype": "",
                            "error_stack": "",
                        }
                    break
                except redis.WatchError as ex:
                    if watch_error_count > 100:
                        raise Exception(
                            "Something fatal happened while creating a new project"
                        ) from ex
                    if watch_error_count > 20:
                        time.sleep(0.1)
                    watch_error_count += 1
                    continue
        return result

    def get_stage_data(self, userid, projectid, stage):
        result_hashmap = self.hashmap(userid, projectid)
        res = self.redis.hget(result_hashmap, stage)
        if res:
            return self.from_JSON(res)
        # if no stage key is found returns None
        return None

    def current_stage_data(self, userid, projectid, stage):
        result_hashmap = self.hashmap(userid, projectid)
        current_stage = self.redis.hget(result_hashmap, "current.stage")
        if stage != current_stage:
            return None
        res = self.get_stage_data(userid, projectid, current_stage)
        if res:
            return res
        # if no stage key is found returns None
        return None

    def revert_stage(self, userid, projectid, from_stage, to_stage):
        result_hashmap = self.hashmap(userid, projectid)
        current_stage = self.redis.hget(result_hashmap, "current.stage")
        if current_stage != from_stage:
            # print("cannot revert ----", current_stage, from_stage)
            return False
        self.redis.hset(result_hashmap, "current.stage", to_stage)
        return True


# used by the backend workers or consumers to process tasks
class PipeTasksConsumer(RedisTasksConsumer):
    # relevant data to get from the hashmap for each of the stages
    stage_data_keys = {
        # `consume:POST` contains the file_name to be processed
        "consume:POST": ["consume:POST"],
        "prepare:POST": ["consume:POST", "prepare:POST"],
        "transform:POST": ["consume:POST", "transform:POST"],
        "build:POST": [
            "consume:POST",
            "prepare:POST",
            "transform:POST",
            "build:GET",
            "build:POST",
        ],
    }

    # keys in the hashmap to store the results of each stage
    stage_result_keys = {
        "consume:POST": "prepare:GET",
        "prepare:POST": "transform:GET",
        "transform:POST": "build:GET",
        "build:POST": "finalconfig:GET",
    }

    def __init__(self, redis_config, func_dict):
        """
        params:
            pool: RedisConnectionPool
            consumer_name: unique fixed consumer name for each consumer
            func_dict: dictionary of functions to process each stage of the pipeline
                {
                    'consume:POST': consume_function,
                    'prepare:POST': prepeare_function,
                    'transform:POST': transform_function,
                    'build:POST': build_function
                }
                each of these functions will accept 2 arguments
                    - stage_data: dictinary containg parsed JSON config recieved from the frontend
                    - file_name: name of the file to process
                returns the config  data for the next stage

                ex: consumer_func(stage_data, file_name) -> config_next_stage
        """
        # print("instantiating pipe task consumer......")
        super().__init__(redis_config, jobq_setting.PIPE_QUEUE)
        self.stage_data = {}
        self.func_dict = func_dict

    @property
    def stage(self):
        return self._item[1]["stage"]

    @property
    def result_hashmap(self):
        return self._item[1]["result_hashmap"]

    def xack(self):
        if self._item:
            self.redis.xack(self.jobq, self.jobq_CG, self.jobid)
            self.stage_data = {}
            self.reset_item()

    def pull_job(self, consumer_name):
        """
        get the active job from the queue and convert JSON config into dict(stage_data)
        """
        while True:
            self.get_item(consumer_name)

            # get associated stage_data from the hashmap associated with the job
            current_stage, pipe_status, current_jobid, *stage_data = self.redis.hmget(
                self.result_hashmap,
                "current.stage",
                "pipe:STATUS",
                f"{self.stage}:jobid",
                *self.stage_data_keys[self.stage],
            )

            # ignore the job, ack it and get the next job
            # check pipe:STATUS
            if (
                (int(pipe_status) != 0)  # type: ignore
                # stage pulled from the job_queue is not the active stage of the pipeline
                or current_stage != self.stage
                # current_jobid supercedes the job pulled from the job queue
                or current_jobid > self.jobid
            ):
                self.xack()
                print("stale items - pull job")
                continue

            for i, key in enumerate(self.stage_data_keys[self.stage]):
                self.stage_data[key] = self.from_JSON(stage_data[i])
            break

    def submit_result(self, result_dict):
        """
        result_dict: dictionary of results to be converted into JSON before storing
        """
        #  use redis pipeline to watch for the hashmap
        #       check if the job is stale
        with self.redis.pipeline() as pipe:
            error_count = 0
            next_stage = self.stage_result_keys[self.stage]
            json_result = self.to_JSON(result_dict)
            while True:
                try:
                    pipe.watch(self.result_hashmap)
                    # after WATCHing, the pipeline is put into immediate execution
                    # mode until we tell it to start buffering commands again.

                    [current_stage, current_jobid, pipe_status] = pipe.hmget(
                        self.result_hashmap,
                        "current.stage",
                        f"{self.stage}:jobid",
                        "pipe:STATUS",
                    )

                    # check if the job is superceded by some other job
                    # check pipe:STATUS
                    if (
                        (int(pipe_status) != 0)
                        # stage pulled from the job_queue is not the active stage of the pipeline
                        or current_stage != self.stage
                        # current_jobid supercedes the job pulled from the job queue
                        or current_jobid != self.jobid
                    ):
                        break

                    # now we can put the pipeline back into buffered mode with MULTI
                    pipe.multi()
                    pipe.hset(self.result_hashmap, next_stage, json_result)
                    pipe.hset(self.result_hashmap, "current.stage", next_stage)
                    # and finally, execute the pipeline (the set command)
                    pipe.execute()
                    # if a WatchError wasn't raised during execution, everything
                    # we just did happened atomically.
                    break
                except redis.WatchError:
                    error_count += 1
                    # print("Watch pipe error")
                    continue

    def freeze_pipe(self, error):
        """
        freeze the pieline due to error in processing
        """
        if not isinstance(error, RedisTasksError):
            error = ConsumerError()
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    pipe.watch(self.result_hashmap)
                    pipe.multi()
                    pipe.hset(self.result_hashmap, "pipe:STATUS", "-1")
                    pipe.hset(self.result_hashmap, "error:TYPE", type(error).__name__)
                    pipe.hset(self.result_hashmap, "error:STACK", error.stack_trace)
                    pipe.execute()
                    break
                except redis.WatchError:
                    watch_error_count += 1
                    continue
                except Exception:
                    print("Error happended while calling freeze_pipe")
                    break

    def run(self, consumer_name):
        while True:
            try:
                print("─────────────────────────   pulling pipe  job   ─────────────────────────")
                self.pull_job(consumer_name)
                print("pulled pipe Job")
                result = self.func_dict[self.stage](self.stage_data)
                print("computed pipe result")
                self.submit_result(result)
                print("submitted pipe result")
            except KeyboardInterrupt:
                break
            except Exception as error:
                self.freeze_pipe(error)
            finally:
                self.xack()


class ModelsTasksProducer(RedisTasksProducer):
    current_stage = "finalconfig:GET"
    models_sorted_set = jobq_setting.MODELS_SORTED_SET

    def __init__(self, redis_config):
        # print("instantiating modle task producer.....")
        super().__init__(redis_config, jobq_setting.MODELS_QUEUE)
        lua_modeljob_submit = """
            local jobid =  redis.call('xadd', KEYS[1], '*', KEYS[2], KEYS[3], KEYS[4], KEYS[5], KEYS[6], KEYS[7], KEYS[8], KEYS[9], KEYS[10], KEYS[11], KEYS[12], KEYS[13], KEYS[14], KEYS[15], KEYS[16], KEYS[17])
            local res1 = redis.call('hset', KEYS[15], KEYS[18], jobid)
            local res2 = redis.call('hset', KEYS[17], jobid, 0)
            local res3 = redis.call('hset', KEYS[11], KEYS[19], KEYS[20])
            return {jobid, res1, res2, res3}
        """
        self.modeljob_add = self.redis.register_script(lua_modeljob_submit.strip())

    @staticmethod
    def get_all_models(modelType):
        if modelType == "regressor":
            return regressors.models
        elif modelType == "classifier":
            return classifiers.models
        elif modelType == "multi_classifier":
            return multi_classifiers.models

    def get_all_model_ids_for_type(self, modelType):
        models = self.get_all_models(modelType)
        return [model.modelid for model in models]

    def get_model_by_ids(self, modelids: List[str], modelType) -> Tuple[List[MLModel], List[str]]:
        models = self.get_all_models(modelType)
        models_to_build = []
        modelids_to_reject = []
        for modelid in modelids:
            m = [model for model in models if model.modelid == modelid]
            if m:
                models_to_build.append(m[0])
            else:
                modelids_to_reject.append(modelid)
        return (models_to_build, modelids_to_reject)

    def submit_model_jobs(
        self,
        user_id,
        project_id,
        optimizeUsing,
        modelType,
        modelids: List[str],
        changed_hparams=None,
        init: bool = False,
    ):  # pylint: disable=unsubscriptable-object
        """
        modelType: one of ["regressor", "classifier", "multi_classifier"]
        each pipeline job is given:
            - modelid
            - modelname
            - model_result_hashmap
            - pipe_result_hashmap (hashmap key where to get the config data, "userid:projectid:.....")
            - field_name (field inside the hashmap key where the config data is stored, "finalconfig:GET")

        also the list of models is added to
            models_sorted_set -> 0 userid:projectid:modelid
        use hashmap 'userid:projectid:CANCELLED:DB_GEN' to track cancellations of jobid
            set(jobids) # use sentinel value 'SENTINEL' while instantiating the set
        """
        if not changed_hparams:
            changed_hparams = {}

        class ModelJobs(BaseModel):
            models_accepted: List[str]
            models_rejected: List[str]
            models_to_build: List[str]
            models_to_ignore: List[str]
            models_to_init: List[str]

        models_accepted: List[MLModel] = []
        # `modelids` for which the `model.modelType` does not match `modelType`
        modelids_rejected: List[str] = []
        models_to_build: List[MLModel] = []
        # if the models is rerun and the status is not in ["DONE", "ERROR", "CANCELLED"]
        # then the models are not added to the job_queue
        models_to_ignore: List[MLModel] = []
        models_to_init: List[MLModel] = []

        models_accepted, modelids_rejected = self.get_model_by_ids(
            [] if not modelids else modelids, modelType
        )

        result = None
        if len(modelids_rejected) > 0:
            return (
                result,
                ModelJobs(
                    models_accepted=[model.modelid for model in models_accepted],
                    models_rejected=modelids_rejected,
                    models_to_build=[],
                    models_to_ignore=[],
                    models_to_init=[],
                ),
            )

        if init:
            print("INFO: request to initialze the pipeline")
            # list of modelids to initialize
            ms = []
            for m in self.get_all_model_ids_for_type(modelType):
                found = False
                for m1 in models_accepted:
                    if m == m1.modelid:
                        found = True
                    break
                if not found:
                    ms.append(m)
            if ms:
                models_to_init, _ = self.get_model_by_ids(ms, modelType)

        if models_accepted or models_to_init:
            # config data used for model building
            pipe_result_hashmap = (
                f"{user_id}:{project_id}:{jobq_setting.PIPE_QUEUE}:{jobq_setting.DB_GEN}"
            )
            # hashmap to store the pipe results
            model_result_hashmap = (
                f"{user_id}:{project_id}:{jobq_setting.MODELS_QUEUE}:{jobq_setting.DB_GEN}"
            )
            # hashmap to store cancelled jobids
            cancelled_hashmap = f"{user_id}:{project_id}:CANCELLED:{jobq_setting.DB_GEN}"

            with self.redis.pipeline() as pipe:
                watch_error_count = 0
                while True:
                    try:
                        pipe.watch(pipe_result_hashmap, model_result_hashmap)
                        current_stage, pipe_status, init_staus = pipe.hmget(
                            pipe_result_hashmap,
                            "current.stage",
                            "pipe:STATUS",
                            "init:STATUS",
                        )

                        pipe_status = int(pipe_status)
                        # pipe_status == -1 (pipeline error)
                        pipe_ok = pipe_status in (0, 1)

                        if pipe_status == 1:
                            job_rerun = True
                        else:
                            job_rerun = False

                        init_staus = 0 if not init_staus else int(init_staus)
                        initialzed = init_staus == 1

                        if init and (initialzed or job_rerun):
                            print("ERROR: trying to reinitialze an initialzed pipeline")
                            break
                        if pipe_ok and (current_stage == self.current_stage):
                            if job_rerun:
                                models_status_keys = []
                                for model in models_accepted:
                                    models_status_keys.append(f"{model.modelid}:STATUS")
                                models_status = pipe.hmget(model_result_hashmap, models_status_keys)
                                models_to_build = []
                                models_to_ignore = []
                                for (status, model) in zip(models_status, models_accepted):
                                    if status in ["DONE", "ERROR", "CANCELLED", "INIT"]:
                                        models_to_build.append(model)
                                    else:
                                        models_to_ignore.append(model)
                                if not models_to_build:
                                    result = None
                                    break
                            else:
                                models_to_build = models_accepted
                                models_to_ignore = []

                            # Submission of model jobs and setting of pipe status are executed as
                            # a single transaction
                            pipe.multi()
                            mss = {}

                            for _, model in enumerate(models_to_build):
                                # print(changed_hparams)
                                if model.modelid not in changed_hparams:
                                    # print(f"{model.modelid} model does not have hyperparams")
                                    hparams = {}
                                else:
                                    hparams = changed_hparams[model.modelid]
                                # print(changed_hparams)
                                self.modeljob_add(
                                    client=pipe,
                                    keys=[
                                        self.jobq,  # 1
                                        "modelid",  # 2
                                        model.modelid,  # 3
                                        "modelname",  # 4
                                        model.modelname,  # 5
                                        "filename",  # 6
                                        model.filename,  # 7
                                        "model_type",  # 8
                                        modelType,  # 9
                                        "pipe_result_hashmap",  # 10
                                        pipe_result_hashmap,  # 11
                                        "field_name",  # 12
                                        "finalconfig:GET",  # 13
                                        "model_result_hashmap",  # 14
                                        model_result_hashmap,  # 15
                                        "cancelled_hashmap",  # 16
                                        cancelled_hashmap,  # 17
                                        f"{model.modelid}:JOBID",  # 18
                                        f"{model.modelid}:HYPERPARAMS",  # 19
                                        self.to_JSON(hparams),  # 20
                                    ],
                                )
                                if init:
                                    mss[f"{user_id}:{project_id}:{model.modelid}"] = 0
                            for model in models_to_build:
                                pipe.hset(model_result_hashmap, f"{model.modelid}:STATUS", "WAIT")
                            if init:
                                for model in models_to_init:
                                    mss[f"{user_id}:{project_id}:{model.modelid}"] = 0
                                    pipe.hset(
                                        model_result_hashmap, f"{model.modelid}:STATUS", "INIT"
                                    )
                                pipe.zadd(self.models_sorted_set, mss)
                                pipe.hset(pipe_result_hashmap, "pipe:STATUS", "1")
                                pipe.hset(pipe_result_hashmap, "init:STATUS", "1")
                                pipe.hset(model_result_hashmap, "model_type", modelType)
                                pipe.hset(model_result_hashmap, "optimizeUsing", optimizeUsing)
                            result = pipe.execute()
                        break
                    except redis.WatchError:
                        if watch_error_count > 100:
                            break
                        if watch_error_count > 20:
                            time.sleep(0.1)
                        watch_error_count += 1
                        continue
                    # Any other exception type is not expected
                    except Exception as ex:
                        print(ex)
                        raise

        return (
            result,
            ModelJobs(
                models_accepted=[model.modelid for model in models_accepted],
                models_rejected=modelids_rejected,
                models_to_build=[model.modelid for model in models_to_build],
                models_to_ignore=[model.modelid for model in models_to_ignore],
                models_to_init=[model.modelid for model in models_to_init],
            ),
        )

    # returns a dictionary of { modelid: status }
    def get_model_status(self, user_id, project_id):
        model_result_hashmap = (
            f"{user_id}:{project_id}:{jobq_setting.MODELS_QUEUE}:{jobq_setting.DB_GEN}"
        )
        list_of_modelid = self.get_models_list(user_id, project_id)
        return dict(
            zip(
                list_of_modelid,
                self.redis.hmget(
                    model_result_hashmap,
                    [f"{modelid}:STATUS" for modelid in list_of_modelid],
                ),
            )
        )

    def model_state_tally(self, user_id, project_id, modelType) -> Optional[Counter]:
        modelids = self.get_all_model_ids_for_type(modelType)
        if modelids is None or len(modelids) == 0:
            return None
        model_result_hashmap = (
            f"{user_id}:{project_id}:{jobq_setting.MODELS_QUEUE}:{jobq_setting.DB_GEN}"
        )
        result = None
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    pipe.watch(model_result_hashmap)
                    pipe.multi()
                    for modelid in modelids:
                        pipe.hget(model_result_hashmap, f"{modelid}:STATUS")
                    result = pipe.execute()
                    break
                except redis.WatchError:
                    if watch_error_count > 5:
                        result = None
                        break
                    watch_error_count += 1
                except Exception as ex:
                    result = None
                    print(ex)
                    break
        if result is None:
            return None
        else:
            return Counter(result)

    def cancel_job(self, user_id, project_id, modelid):
        model_result_hashmap = (
            f"{user_id}:{project_id}:{jobq_setting.MODELS_QUEUE}:{jobq_setting.DB_GEN}"
        )
        cancelled_hashmap = f"{user_id}:{project_id}:CANCELLED:{jobq_setting.DB_GEN}"
        result = None
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    pipe.watch(model_result_hashmap, cancelled_hashmap)
                    current_jobid, current_status = pipe.hmget(
                        model_result_hashmap, f"{modelid}:JOBID", f"{modelid}:STATUS"
                    )
                    print(f"current_jobid for cancellation = {current_jobid} {current_status}")
                    if current_status in ["WAIT", "RUNNING"]:
                        pipe.multi()
                        pipe.hget(model_result_hashmap, f"{modelid}:STATUS")
                        pipe.hset(cancelled_hashmap, current_jobid, 1)
                        pipe.hset(model_result_hashmap, f"{modelid}:STATUS", "TRYCANCEL")
                        result = pipe.execute()
                        result = (2, result)
                    elif current_status == "CANCELLED":
                        result = (1, [])
                    else:
                        result = (0, [])
                    break
                except redis.WatchError:
                    if watch_error_count > 5:
                        result = None
                        break
                    watch_error_count += 1
                except Exception as ex:
                    result = None
                    print(ex)
                    break
        return result

    # returns a dictionary of { modelid: data }
    def get_model_data(self, user_id, project_id, list_of_modelid, include_error=False):
        model_result_hashmap = (
            f"{user_id}:{project_id}:{jobq_setting.MODELS_QUEUE}:{jobq_setting.DB_GEN}"
        )
        res = self.redis.hmget(
            model_result_hashmap,
            (
                [f"{modelid}:DATA" for modelid in list_of_modelid]
                + [f"{modelid}:STATUS" for modelid in list_of_modelid]
                + [f"{modelid}:ERROR" for modelid in list_of_modelid]
                + ["optimizeUsing", "model_type"]
            ),
        )
        model_type = res[-1]
        optimizeUsing = res[-2]
        items = {}
        lm = len(list_of_modelid)
        for i, modelid in enumerate(list_of_modelid):
            items[f"{modelid}:DATA"] = res[i]
            # print(res[i])
            items[f"{modelid}:STATUS"] = res[lm + i]
            # print(res[lm + i])
            items[f"{modelid}:ERROR"] = res[(2 * lm) + i]
        # pprint(items)
        models = []
        for modelid in list_of_modelid:
            if items[f"{modelid}:DATA"]:
                model = self.from_JSON(items[f"{modelid}:DATA"])
            else:
                model = {}
            model["id"] = modelid
            model["status"] = items[f"{modelid}:STATUS"]
            # do not send error trace to frontend. Use it internally for debugging.
            if include_error and items[f"{modelid}:STATUS"] == "ERROR":
                model["ERROR"] = items[f"{modelid}:ERROR"]
            else:
                model["ERROR"] = ""
            models.append(model)
        # pprint(models)
        return {
            "modelType": model_type,
            "optimizeUsing": optimizeUsing,
            "models": models,
        }

    def get_models_list(self, user_id, project_id):
        models = self.redis.zrangebylex(
            self.models_sorted_set,
            f"[{user_id}:{project_id}:",
            f"[{user_id}:{project_id}:\xff",
        )
        if not models:
            return models
        return [model.split(sep=":")[2] for model in models]


class ModelsTasksConsumer(RedisTasksConsumer):
    """
    model_result_hashmap - {userid}:{projectid}:{models_queue}:{DB_GEN}
        modelid:DATA -> final result
        modelid:STATUS -> one of ["INIT", "WAIT", "RUNNING", "ERROR", "DONE","TRYCANCEL", "CANCELLED"]
        modelid:JOBID
        modelid:ERROR -> error message if any
        modelid:PICKLE -> path to pickled model
    """

    def __init__(self, redis_config, model_function_table, save_pickled_model, pickle_config):
        super().__init__(redis_config, jobq_setting.MODELS_QUEUE)
        self.model_function_table = model_function_table
        # function to execute for saving pickled model
        self.save_pickled_model = save_pickled_model
        # contains the folder where the model is to be saved
        self.pickle_config = pickle_config

    def pull_job(self, consumer_name):
        while True:
            self.get_item(consumer_name)
            model_result_hashmap = self._item[1]["model_result_hashmap"]
            cancelled_hashmap = self._item[1]["cancelled_hashmap"]
            modelid = self._item[1]["modelid"]

            result = None
            with self.redis.pipeline() as pipe:
                watch_error_count = 0
                while True:
                    try:
                        pipe.watch(model_result_hashmap)
                        last_jobid = pipe.hget(model_result_hashmap, f"{modelid}:JOBID")
                        cancelled_status = int(pipe.hget(cancelled_hashmap, self.jobid))

                        # ignore the job, ack it and get the next job
                        # print(self._item[1]["cancelled_hashmap"])
                        if last_jobid and last_jobid > self.jobid:
                            pipe.multi()
                            pipe.xack(self.jobq, self.jobq_CG, self.jobid)
                            pipe.execute()
                            print("stale items - pull model job")
                            self.reset_item()
                        elif cancelled_status != 0:
                            pipe.multi()
                            pipe.hset(model_result_hashmap, f"{modelid}:STATUS", "CANCELLED")
                            pipe.xack(self.jobq, self.jobq_CG, self.jobid)
                            pipe.execute()
                            print(f"cancelled job: {self.jobid}")
                            self.reset_item()
                        else:
                            pipe.multi()
                            pipe.hget(
                                self._item[1]["pipe_result_hashmap"],
                                self._item[1]["field_name"],
                            )
                            pipe.hset(model_result_hashmap, f"{modelid}:STATUS", "RUNNING")
                            pipe.hget(
                                self._item[1]["pipe_result_hashmap"],
                                f"{self._item[1]['modelid']}:HYPERPARAMS",
                            )
                            pipe.hget(
                                self._item[1]["model_result_hashmap"],
                                f"{self._item[1]['modelid']}:DATA",
                            )
                            ret = pipe.execute()
                            # debug("************Result********:", ret[3])
                            if not ret[3]:
                                model_result = {}
                            else:
                                model_result = self.from_JSON(ret[3])
                            result = {
                                "jobid": self.jobid,
                                "modelid": self._item[1]["modelid"],
                                "modelname": self._item[1]["modelname"],
                                "model_type": self._item[1]["model_type"],
                                "model_result_hashmap": self._item[1]["model_result_hashmap"],
                                "data": self.from_JSON(ret[0]),
                                "hyper_params": self.from_JSON(ret[2]),
                                "model_result_dict": model_result,
                            }
                        break
                    except redis.WatchError:
                        if watch_error_count > 100:
                            break
                        if watch_error_count > 20:
                            time.sleep(0.1)
                        watch_error_count += 1
                        continue
                    # Any other exception type is not expected
                    except Exception as ex:
                        raise ex
            if result:
                break
        return result

    def submit_result(
        self,
        model_result_hashmap,
        jobid,
        modelid,
        modelstatus,
        result_dict,
        model_to_pickle,
        error_msg,
        cancelled_hashmap,
    ):
        """
        modelstatus -> one of ["DONE", "ERROR"]
        """
        projectid = model_result_hashmap.split(sep=":")[1]

        folder_name = projectid
        file_path = f"{folder_name}/{datetime.datetime.now().strftime('%s')}___{modelid}.pkl.zip"
        modelid = self._item[1]["modelid"]

        with self.redis.pipeline() as pipe:
            error_count = 0
            while True:
                try:
                    pipe.watch(model_result_hashmap, cancelled_hashmap)
                    # after WATCHing, the pipeline is put into immediate execution
                    # mode until we tell it to start buffering commands again.
                    last_jobid = pipe.hget(model_result_hashmap, f"{modelid}:JOBID")
                    if last_jobid and last_jobid > self.jobid:
                        # do the acking in the run loop
                        # self.redis.xack(self.jobq, self.jobq_CG, self.jobid)
                        print("ignoring job result - superceded")
                        break
                    cancelled_status = int(pipe.hget(cancelled_hashmap, self.jobid))
                    if cancelled_status != 0:
                        modelstatus = "CANCELLED"
                        result_dict = {}
                    # now we can put the pipeline back into buffered mode with MULTI
                    pipe.multi()
                    pipe.hset(model_result_hashmap, f"{modelid}:JOBID", jobid)
                    pipe.hset(model_result_hashmap, f"{modelid}:STATUS", modelstatus)
                    if modelstatus == "ERROR":
                        pipe.hset(model_result_hashmap, f"{modelid}:ERROR", error_msg)
                    else:
                        pipe.hset(
                            model_result_hashmap,
                            f"{modelid}:DATA",
                            self.to_JSON(result_dict),
                        )
                        pipe.hset(model_result_hashmap, f"{modelid}:PICKLE", file_path)
                    # and finally, execute the pipeline (the set command)
                    pipe.execute()
                    # debug("*****###****", result_dict["hp_results"])
                    # if a WatchError wasn't raised during execution, everything
                    # we just did happened atomically.

                    # pickle the model
                    if modelstatus == "DONE":
                        stored = self.pickle_model(folder_name, file_path, model_to_pickle)
                        if not stored:
                            pipe.multi()
                            pipe.hset(
                                model_result_hashmap,
                                f"{modelid}:ERROR",
                                f"error during pickling of the model {modelid}",
                            )
                            pipe.hset(model_result_hashmap, f"{modelid}:STATUS", "ERROR")
                            pipe.execute()
                    break
                except redis.WatchError as ex:
                    if error_count > 100:
                        raise Exception(
                            "Something fatal happened while submitting result of model job"
                        ) from ex
                    if error_count > 20:
                        time.sleep(0.1)
                    error_count += 1
                    continue

    def pickle_model(self, folder_name, file_path, model_to_pickle):
        try:
            zipped_pickled_bytes = self.pickle_gzip(model_to_pickle)
            return self.save_pickled_model(
                self.pickle_config["MODELS_VOLUME"], folder_name, file_path, zipped_pickled_bytes
            )
        except Exception as ex:
            print(traceback.format_exc())
            print(ex)
            return False

    def run(self, consumer_name):
        while True:
            job = None
            cancelled_hashmap = None
            try:
                print("─────────────────────────   pulling model job   ─────────────────────────")
                job = self.pull_job(consumer_name)
                print(
                    "pulled model Job... processing",
                    "jobid = ",
                    job["jobid"],
                    "modelname = ",
                    job["modelname"],
                )
                # time.sleep(10)
                # job["data"] -> dict of finalconfig:GET data

                config = {
                    "modelid": job["modelid"],
                    "modelname": job["modelname"],
                    "model_type": job["model_type"],
                    "data": job["data"],
                    "hyper_params": job["hyper_params"],
                    "hp_results": job["model_result_dict"].get("hp_results", None),
                }
                # debug(config["hp_results"], "**********", job["model_result_dict"]["hp_results"])
                cancelled_hashmap = cancelled_hashmap = self._item[1]["cancelled_hashmap"]
                # (result_dict, model_to_pickle) = self.model_func_dict[job["modelid"]](config)
                (result, error_msg, cancelled, exception) = execute_task(
                    self.model_function_table[job["modelid"]],
                    config,
                    self.redis_config,
                    cancelled_hashmap,
                    self.jobid,
                )
                if cancelled:
                    if exception:
                        modelstatus = "ERROR"
                    else:
                        modelstatus = "CANCELLED"
                else:
                    if exception:
                        modelstatus = "ERROR"
                    else:
                        modelstatus = "DONE"
                if (not exception) and (not cancelled) and result:
                    (result_dict, model_to_pickle) = result
                else:
                    (result_dict, model_to_pickle) = ({}, {})
            except KeyboardInterrupt:
                break
            except Exception as error:
                result_dict = {}
                model_to_pickle = {}
                modelstatus = "ERROR"
                error_msg = traceback.format_exc()
                if not (isinstance(error, KeyboardInterrupt)):
                    print(error_msg)
            finally:
                if self._item and job and cancelled_hashmap:
                    self.submit_result(
                        job["model_result_hashmap"],
                        job["jobid"],
                        job["modelid"],
                        modelstatus,
                        result_dict,
                        model_to_pickle,
                        error_msg,
                        cancelled_hashmap,
                    )
                    self.redis.xack(self.jobq, self.jobq_CG, job["jobid"])
                    print(
                        "submitted model result",
                        "jobid = ",
                        job["jobid"],
                        "modelname = ",
                        job["modelname"],
                    )
                job = None


User = namedtuple("User", ["id", "password"])


class Application:
    def __init__(self, redis_config):
        """
        pool: redis ConnectionPool
        projects_set: (sorted set), lexicographical queries
            0 user_id1:project_id1:unix_time1:project_name1
            0 user_id2:project_id2:unix_time2:project_name2
        current_project: (hashmap)
            user_id -> project_id
        uploads_set (sorted set), lexicographical queries
            0 user_id:{timestamp}__{filename}
        """
        self.redis_config = redis_config
        self.redis = redis.Redis(connection_pool=redis.ConnectionPool(**redis_config))
        self.projects_set = jobq_setting.PROJECTS_SORTED_SET
        self.current_project = jobq_setting.CURRENT_PROJECT
        self.uploads_set = jobq_setting.UPLOADS_SORTED_SET
        self.projects_desc_hashmap = jobq_setting.PROJECTS_DESC_HASHMAP
        # check if we are able to talk to redis
        assert self.redis.ping(), "Can not Ping redis server"

    @staticmethod
    def digest(string):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()

    def add_project(self, user_id, project_name, project_desc):
        with self.redis.pipeline() as pipe:
            watch_error_count = 0
            while True:
                try:
                    project_id = uuid.uuid4().hex
                    timestamp = datetime.datetime.now().strftime("%s")
                    pipe.watch(self.projects_set)
                    res = pipe.zlexcount(
                        self.projects_set,
                        f"[{user_id}:{project_id}:",
                        f"[{user_id}:{project_id}:\xff",
                    )
                    if int(res) == 0:
                        pipe.multi()
                        pipe.zadd(
                            self.projects_set,
                            {f"{user_id}:{project_id}:{timestamp}:{project_name}": 0},
                        )
                        pipe.hset(
                            self.projects_desc_hashmap,
                            f"{user_id}:{project_id}",
                            project_desc,
                        )
                        pipe.execute()
                        return project_id
                    # generate a different project_id
                    continue
                except redis.WatchError as ex:
                    if watch_error_count > 100:
                        print(ex)
                        raise Exception(
                            "Something fatal happened while creating a new project"
                        ) from ex
                    if watch_error_count > 20:
                        time.sleep(0.1)
                    watch_error_count += 1
                    continue

    def projects_list(self, user_id):
        projects = self.redis.zrangebylex(
            self.projects_set,
            f"[{user_id}:",
            f"[{user_id}:\xff",
        )
        if not projects:
            return projects
        projects = [proj.split(sep=":") for proj in projects]
        projects_desc = self.redis.hmget(
            self.projects_desc_hashmap, [f"{p[0]}:{p[1]}" for p in projects]
        )
        # print(projects_desc)
        projects = [(p + [projects_desc[i]]) for i, p in enumerate(projects)]
        # print(projects)
        return sorted(
            (
                {
                    "projectid": p[1],
                    "projectname": p[3],
                    "timestamp": datetime.datetime.utcfromtimestamp(int(p[2])).isoformat(),
                    "projectdesc": p[4],
                }
                for p in filter(lambda p: p[0] == user_id, projects)
            ),
            key=lambda x: x["timestamp"],
            reverse=True,
        )

    def set_current_projectid(self, user_id, project_id):
        return self.redis.hset(self.current_project, user_id, project_id)

    def get_current_projectid(self, user_id):
        """returns None if no project set, otherwise returns the projectid"""
        return self.redis.hget(self.current_project, user_id)

    def verify_projectid(self, userid, projectid):
        res = projects = self.redis.zrangebylex(
            self.projects_set,
            f"[{userid}:{projectid}:",
            f"[{userid}:{projectid}:\xff",
        )
        if not res:
            return None
        res = projects[0].split(sep=":")
        return res[3], datetime.datetime.utcfromtimestamp(int(res[2])).isoformat()

    @staticmethod
    def timestamp_file_name(file_name):
        return f"{datetime.datetime.now().strftime('%s')}___{file_name}"

    def set_upload(self, user_id, file_with_timestamp):
        """
        check that file_name does not contain `:` character
            params:
                file_with_timestamp: "{timestamp}__test_xyz.csv"
        """
        return self.redis.zadd(self.uploads_set, {f"{user_id}:{file_with_timestamp}": 0})

    def get_uploads(self, user_id):
        uploads = self.redis.zrangebylex(
            self.uploads_set,
            f"[{user_id}:",
            f"[{user_id}:\xff",
        )
        if not uploads:
            return uploads
        return sorted(
            (
                {
                    "timestamp": datetime.datetime.utcfromtimestamp(
                        int(p[1].split("___", maxsplit=1)[0])
                    ).isoformat(),
                    "filename": p[1].split("___", maxsplit=1)[1],
                    "filepath": f"{user_id}/{p[1]}",
                }
                for p in filter(lambda p: p[0] == user_id, (up.split(sep=":") for up in uploads))
            ),
            key=lambda p: p["timestamp"],
            reverse=True,
        )
