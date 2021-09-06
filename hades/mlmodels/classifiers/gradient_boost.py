from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from hades.mlmodels.utils import (
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

from ...mlmodels.utils import empty_choices


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

    clf_fit, clf_results = gridSearchGradientBoostingClf(X_train, Y_train, config, model_config)

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
            possible_model_params=empty_choices(possible_param_grid),
        ),
        clf_fit,
    )


def gridSearchGradientBoostingClf(X, Y, config, model_config=None):
    gsClf = GridSearchCV(
        GradientBoostingClassifier(n_iter_no_change=5, tol=0.01, random_state=0),
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
        "loss": {"default": "deviance", "possible_str": ["deviance", "exponential"]},
        "learning_rate": {
            "default": 0.1,
            "possible_float": [0.0001, 1000],
        },
        "n_estimators": {
            "default": 100,
            "possible_int": [10, 1000],
        },
        "criterion": {"default": "friedman_mse", "possible_str": ["friedman_mse", "mse"]},
        "max_depth": {"default": None, "possible_int": [1, config["data"]["rows"]]},
        "min_samples_split": {"default": 2, "possible_int": [2, config["data"]["rows"]]},
        "min_samples_leaf": {"default": 1, "possible_int": [1, config["data"]["rows"]]},
        "max_features": {"default": None, "possible_str": [None, "auto", "sqrt", "log2"]},
    }
    default_hp_grid = {
        "n_estimators": [50, 100],
        "max_features": ["auto", "sqrt"],
    }
    return (possible_param_grid, default_hp_grid)
