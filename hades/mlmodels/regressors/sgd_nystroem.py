from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from hades.mlmodels.utils import (
    convert_cvresults_tolist,
    deliverformattedResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultRegressor,
    splitTrainTestdataset,
)

from ...mlmodels.utils import empty_choices


def build(confign):
    config = confign["data"]
    finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)

    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    possible_param_grid, default_hp_grid = paramlist(confign)
    model_config = confign["hyper_params"]
    if not model_config:
        model_config = default_hp_grid

    reg_fit, reg_results = gridSearchSGDRegressor(X_train, Y_train, config, model_config)

    if not confign["hp_results"]:
        confign["hp_results"] = [
            {
                "hp_grid": model_config,
                "result": reg_results,
            },
        ]
    else:
        hp_result = {
            "hp_grid": model_config,
            "result": reg_results,
        }
        confign["hp_results"].append(hp_result)

    # print(reg.best_params_, reg.best_score_)
    # print(reg_fit["feature_selection"].get_support())
    # print(reg_fit["feature_selection"].threshold_)
    # print(reg_fit["reg"].coef_)

    Y_pred = reg_fit.predict(X_test)
    Y_score = reg_fit.score(X_test, Y_test)
    metricResult = metricResultRegressor(Y_test, Y_pred, Y_score)
    # plotPredictedVsTrueCurve(Y_pred, Y_test, X_test, modelName)

    return (
        deliverformattedResult(
            config,
            metricResult,
            Y_pred,
            Y_test,
            grid_results=reg_results,
            hp_results=confign["hp_results"],
            possible_model_params=empty_choices(possible_param_grid),
        ),
        reg_fit,
    )


def gridSearchSGDRegressor(X, Y, config, model_config=None):
    steps = [
        # ("feature_selection", SelectFromModel(LassoCV(), "median")),
        ("feature_map_Nystroem", Nystroem(n_components=300)),
        ("reg", SGDRegressor(random_state=100, max_iter=1000)),
    ]
    make_pipeline = Pipeline(steps)
    gsreg = GridSearchCV(
        make_pipeline,
        param_grid=model_config,
        cv=config["data"]["cv"]["folds"],
    )
    gsreg_fit = gsreg.fit(X, Y)
    gsreg_fit_estimator = gsreg_fit.best_estimator_
    gsreg_results = {
        "cvresult_list": convert_cvresults_tolist(gsreg_fit.cv_results_),
        "mean_test_score": gsreg_fit.cv_results_["mean_test_score"].tolist(),
        "params": gsreg_fit.cv_results_["params"],
        # gsClf_fit.best_estimator_,
        "best_score": round(gsreg_fit.best_score_, 2),
        "best_params": list(zip(gsreg_fit.best_params_.keys(), gsreg_fit.best_params_.values())),
        # "scorer_function": str(gsClf_fit.scorer_),
        # gsClf_fit.best_index_,
    }
    return gsreg_fit_estimator, gsreg_results


def paramlist(confign):
    config = confign["data"]
    possible_param_grid = {
        "reg__loss": {
            "default": "squared_loss",
            "possible_str": ["squared_loss", "huber", "epsilon_insensitive"],
        },
        "reg__penalty": {
            "default": "l2",
            "possible_str": ["l2", "l1", "elasticnet"],
        },
        "reg__alpha": {
            "default": 0.0001,
            "possible_str": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        },
        "reg__max_iter": {
            "default": 1000,
            "possible_str": [1000, 10000],
        },
        "reg__learning_rate": {
            "default": "invscaling",
            "possible_list": ["constant", "invscaling", "adaptive", "optimal"],
        },
        "reg__warm_start": {"default": False, "possible_str": [True, False]},
    }
    default_hp_grid = {
        "feature_map_Nystroem__gamma": [0.2, 0.5],
        "reg__alpha": [0.0001, 0.01],
    }
    return (possible_param_grid, default_hp_grid)
