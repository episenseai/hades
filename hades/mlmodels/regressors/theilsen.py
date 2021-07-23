from sklearn.linear_model import TheilSenRegressor
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

    reg_fit, reg_results = gridSearchTheilSenRegressor(X_train, Y_train, config, model_config)

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


def gridSearchTheilSenRegressor(X, Y, config, model_config=None):
    steps = [("reg", TheilSenRegressor(random_state=100))]
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
        "reg__max_iter": {
            "default": 300,
            "possible_list": [300, 1000],
        },
        "reg__fit_intercept": {"default": True, "possible_str": [True, False]},
    }
    default_hp_grid = {"reg__fit_intercept": [True, False]}
    return (possible_param_grid, default_hp_grid)
