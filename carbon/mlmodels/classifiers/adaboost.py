from devtools import debug
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from carbon.mlmodels.utils import (
    convert_cvresults_tolist,
    deliverformattedResultClf,
    deliverRoCResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultMultiClassifier,
    rocCurveforClassDecisionFunction,
    splitTrainTestdataset,
)


def build(confign):
    config = confign["data"]
    finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    Y = Y.astype(str)
    catClasses = Y.unique()

    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)

    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    possible_param_grid, default_hp_grid = paramlist(confign)
    model_config = confign["hyper_params"]
    if not model_config:
        model_config = default_hp_grid

    clf_fit, clf_results = gridSearchAdaBoostClf(X_train, Y_train, config, model_config)

    if not confign["hp_results"]:
        confign["hp_results"] = [
            {
                "hp_grid": model_config,
                "result": clf_results,
            },
        ]
    else:
        hp_result = {
            "hp_grid": model_config,
            "result": clf_results,
        }
        confign["hp_results"].append(hp_result)

    # Plot of a ROC curve for a specific class
    fpr, tpr, roc_auc, Y_pred, Y_score = rocCurveforClassDecisionFunction(
        X_train, X_test, Y_train, Y_test, catClasses, clf_fit
    )
    # print((Y_test, Y_pred))
    confusion = confusion_matrix(Y_test, Y_pred)

    metricResult = metricResultMultiClassifier(Y_test, Y_pred, Y_score)
    # plotRoCCurve(catClasses, fpr, tpr, roc_auc)

    roc = deliverRoCResult(catClasses, fpr, tpr, roc_auc)
    return (
        deliverformattedResultClf(
            config,
            catClasses,
            metricResult,
            confusion,
            roc,
            grid_results=clf_results,
            hp_results=confign["hp_results"],
            possible_model_params=possible_param_grid,
        ),
        clf_fit,
    )


def gridSearchAdaBoostClf(X, Y, config, model_config=None):
    gsClf = GridSearchCV(
        AdaBoostClassifier(
            random_state=0,
        ),
        param_grid=model_config,
        cv=config["data"]["cv"]["folds"],
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    gsclf_results = {
        "cvresult_list": convert_cvresults_tolist(gsClf_fit.cv_results_),
        "mean_test_score": gsClf_fit.cv_results_["mean_test_score"].tolist(),
        "params": gsClf_fit.cv_results_["params"],
        # gsClf_fit.best_estimator_,
        "best_score": round(gsClf_fit.best_score_, 2),
        "best_params": list(zip(gsClf_fit.best_params_.keys(), gsClf_fit.best_params_.values())),
        # "scorer_function": str(gsClf_fit.scorer_),
        # gsClf_fit.best_index_,
    }

    return gsClf_fit_estimator, gsclf_results


def paramlist(confign):
    config = confign["data"]
    possible_param_grid = {
        "n_estimators": {
            "default": 50,
            "possible_list": list(range(50, 500, 50)),
        },  # [min,max]
        "learning_rate": {"default": 1, "possible_list": [0.5, 0.75, 1, 1.25, 1.5, 2]},
        "algorithm": {"default": "SAMME.R", "possible_str": ["SAMME", "SAMME.R"]},
    }
    default_hp_grid = {"n_estimators": [50, 100], "learning_rate": [0.1, 1, 2]}
    return (possible_param_grid, default_hp_grid)
