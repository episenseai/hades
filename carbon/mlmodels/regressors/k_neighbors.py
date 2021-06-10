from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# from Models.config import config1, config2, config3
# from datetime import datetime
# from pprint import pprint


def build(confign):
    config = confign["data"]
    finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)

    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    # print("start", datetime.now())
    reg, reg_fit, reg_results = gridSearchkNeighborsRegressor(X_train, Y_train, config)
    # print("end", datetime.now())

    # print(reg.best_params_, reg.best_score_)
    # print(reg_fit["feature_selection"].get_support())
    # print(reg_fit["feature_selection"].threshold_)
    # print(reg_fit["reg"].coef_)

    Y_pred = reg_fit.predict(X_test)
    Y_score = reg_fit.score(X_test, Y_test)
    metricResult = metricResultRegressor(Y_test, Y_pred, Y_score)
    # plotPredictedVsTrueCurve(Y_pred, Y_test, X_test, modelName)

    return (
        deliverformattedResult(config, metricResult, Y_pred, Y_test, grid_results=reg_results),
        reg_fit,
    )


def gridSearchkNeighborsRegressor(X, Y, config):
    steps = [
        ("scalar", StandardScaler()),
        ("feature_selection", SelectFromModel(LassoCV(), "median")),
        ("reg", KNeighborsRegressor()),
    ]
    make_pipeline = Pipeline(steps)
    gsreg = GridSearchCV(
        make_pipeline,
        param_grid={
            "reg__weights": ["uniform", "distance"],
            "reg__algorithm": ["auto", "brute"],
        },
        cv=config["data"]["cv"]["folds"],
    )
    gsreg_fit = gsreg.fit(X, Y)
    gsreg_fit_estimator = gsreg_fit.best_estimator_
    # print(gsreg_fit.best_params_, gsreg_fit.best_score_)
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
    return gsreg, gsreg_fit_estimator, gsreg_results


# pprint(build_model(config3))
