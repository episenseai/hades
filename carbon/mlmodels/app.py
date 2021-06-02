from multiprocessing import Process

from ..store.backend.redis import ModelsTasksConsumer
from .classifiers import adaboost, bagging, bernoulli_nb
from .classifiers import decision_tree as decision_tree_classifier
from .classifiers import gradient_boost
from .classifiers import linear_sv as linear_sv_classifier
from .classifiers import logistic_regression
from .classifiers import mlp as mlp_classifier
from .classifiers import multinomial_nb, passive_aggressive, ridge, sgd
from .config import mlmodels_config
from .regressors import decision_tree as decision_tree_regressor
from .regressors import k_neighbors
from .regressors import linear_sv as linear_sv_regressor
from .regressors import mlp as mlp_regressor
from .regressors import nu_sv, radius_neighbors, sgd_nystroem, theilsen

# funcion to process model job workers
func_dict = {
    # classifier models
    "24ee24ed-6174-4a79-bf53-215d6fbcf680": adaboost.build,
    "fec5dfd9-335f-4f3c-942f-2965a00fcdbe": bagging.build,
    "4cbc5b86-2143-4cc7-9dcb-7c3700ec4db3": bernoulli_nb.build,
    "6bb167c7-fd88-4fc1-8cc9-5005b463a6b4": decision_tree_classifier.build,
    "f4acfa46-5063-4a20-bc8c-9c44f1be9d66": gradient_boost.build,
    "37c1d7c6-3914-4924-bbac-b057d8e2247b": linear_sv_classifier.build,
    "cf396a47-13d0-4ddd-b786-7b94a4852f72": logistic_regression.build,
    "bb4ae778-4229-429d-bcbe-ed76872f16a1": mlp_classifier.build,
    "2dc4beda-3b36-4421-a014-87f9e0bfa778": multinomial_nb.build,
    "88f1c34d-bfee-49b1-91e6-d3ea66347c6e": passive_aggressive.build,
    "7116745c-9f8d-458d-b7d6-c2aae90790b3": ridge.build,
    "ae811751-603b-4af4-83bf-39a9eb7bf77f": sgd.build,
    # regressor models
    "4be01ae8-5e00-497c-903c-214ded1f1724": decision_tree_regressor.build,
    "83e55067-aa8e-4d4b-961e-bef5218096c9": k_neighbors.build,
    "60c1f21c-7bbb-4119-981a-04a42b1e54cf": linear_sv_regressor.build,
    "259c7cf1-33fb-4768-ba88-2d2fac348c95": mlp_regressor.build,
    "7cc05ccd-580e-4233-8604-e0d5ea523da5": nu_sv.build,
    "ede524b9-de83-469b-9b64-a09c9d4aac9e": radius_neighbors.build,
    "6344633f-f4b3-4b1b-9203-32b924c78691": sgd_nystroem.build,
    "a10dd9dc-cda3-4aa5-b9f3-ac6d26140bd1": theilsen.build,
}

# list of workers
allowed_model_workers = mlmodels_config.workers.worker_names[
    : (min(mlmodels_config.workers.num_workers, len(mlmodels_config.workers.worker_names)))
]

ps = []


# spawn this worker func onto a process
def models_consumer_func(worker):
    import warnings

    warnings.simplefilter("ignore")

    consumer = ModelsTasksConsumer(mlmodels_config.redis.dict(), func_dict)
    try:
        consumer.run(worker)
    except Exception as ex:
        if not (isinstance(ex, KeyboardInterrupt)):
            print("-------------Exception happened in model worker: ", worker)
            print(ex)
        # import traceback
        # print(traceback.format_exc())


def spawn_model_workers():

    for worker in allowed_model_workers:
        p = Process(target=models_consumer_func, args=(worker,))
        p.start()
        ps.append(p)


def join_model_workers():
    for p in ps:
        p.join()


# SIGKILL process (no cleanup)
def kill_model_workers():
    for i, p in enumerate(ps):
        p.kill()
        p.join()
    print("============ KILLED MODEL WORKERS (MLMODELS) ============")
