from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hades.mlmodels.utils import (
    convert_cvresults_tolist,
    deliverformattedResultClf,
    deliverRoCResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultMultiClassifier,
    rocCurveforClassPredictProba,
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

    clf_fit, clf_results = gridSearchLogisticRegressionClf(X_train, Y_train, config, model_config)

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
    fpr, tpr, roc_auc, Y_pred, Y_score = rocCurveforClassPredictProba(
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
            possible_model_params=empty_choices(possible_param_grid),
        ),
        clf_fit,
    )


def gridSearchLogisticRegressionClf(X, Y, config, model_config=None):
    steps = [
        ("scalar", StandardScaler()),
        (
            "clf",
            LogisticRegression(multi_class="multinomial", max_iter=500),
        ),
    ]
    make_pipeline = Pipeline(steps)
    gsClf = GridSearchCV(
        make_pipeline,
        param_grid=model_config,
        cv=config["data"]["cv"]["folds"],
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    # print(gsClf_fit.best_params_, gsClf_fit.best_score_)
    # print(gsClf_fit.cv_results_)
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
        "clf__penalty": {"default": "l2", "possible_str": ["l1", "l2", "elasticnet", "none"]},
        "clf__C": {
            "default": 1.0,
            "possible_list": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        "clf__solver": {
            "default": "lbfgs",
            "possible_list": ["newton-cg", "sag", "saga", "lbfgs"],
        },
        "clf__warm_start": {"default": False, "possible_str": [True, False]},
    }
    default_hp_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    return (possible_param_grid, default_hp_grid)
