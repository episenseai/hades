from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from carbon.mlmodels.utils import (
    convert_cvresults_tolist,
    deliverformattedResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultRegressor,
    splitTrainTestdataset,
)


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

    reg_fit, reg_results = gridSearchDecisionTreeRegressor(X_train, Y_train, config, model_config)

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
            possible_model_params=possible_param_grid,
        ),
        reg_fit,
    )


def gridSearchDecisionTreeRegressor(X, Y, config, model_config=None):
    steps = [
        ("feature_selection", SelectFromModel(LassoCV(), "median")),
        ("reg", DecisionTreeRegressor(random_state=100)),
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
        "best_score": round(gsreg_fit.best_score_, 2),
        "best_params": list(zip(gsreg_fit.best_params_.keys(), gsreg_fit.best_params_.values())),
    }
    # print(gsreg_fit.cv_results_)
    return gsreg_fit_estimator, gsreg_results


def paramlist(confign):
    config = confign["data"]
    possible_param_grid = {
        "reg__criterion": {
            "default": "mse",
            "possible_str": ["mse", "friedman_mse", "mae", "poisson"],
        },  # [min,max]
        "reg__splitter": {"default": "best", "possible_str": ["best", "random"]},
        "reg__max_depth": {"default": None, "possible_int": [1, config["data"]["rows"]]},
        "reg__min_samples_split": {"default": 2, "possible_int": [2, config["data"]["rows"]]},
        "reg__min_samples_leaf": {"default": 1, "possible_int": [1, config["data"]["rows"]]},
        "reg__max_features": {"default": "auto", "possible_str": ["auto", "sqrt", "log2"]},
        "reg__max_leaf_nodes": {"default": None, "possible_int": [1, config["data"]["rows"]]},
    }

    default_hp_grid = {
        "reg__min_samples_split": list(range(2, 32, 10)),
        "reg__max_features": list(range(2, len(finalFeatureListGenerator(config)), 10)),
    }
    return (possible_param_grid, default_hp_grid)
