from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from carbon.mlmodels.utils import (
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
    modelName = "K Neighbors Regressor"
    config = confign["data"]
    finalFeatureSet = finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)

    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    # print("start", datetime.now())
    reg, reg_fit = gridSearchTheilSenRegressor(X_train, Y_train, config)
    # print("end", datetime.now())

    # print(reg.best_params_, reg.best_score_)
    # print(reg_fit["feature_selection"].get_support())
    # print(reg_fit["feature_selection"].threshold_)
    # print(reg_fit["reg"].coef_)

    Y_pred = reg_fit.predict(X_test)
    Y_score = reg_fit.score(X_test, Y_test)
    metricResult = metricResultRegressor(Y_test, Y_pred, Y_score)
    # plotPredictedVsTrueCurve(Y_pred, Y_test, X_test, modelName)

    return deliverformattedResult(config, metricResult, Y_pred, Y_test), reg_fit


def gridSearchTheilSenRegressor(X, Y, config):
    steps = [("reg", TheilSenRegressor(random_state=100))]
    make_pipeline = Pipeline(steps)
    gsreg = GridSearchCV(
        make_pipeline,
        param_grid={"reg__fit_intercept": [True, False]},
        cv=config["data"]["cv"]["folds"],
    )
    gsreg_fit = gsreg.fit(X, Y)
    gsreg_fit_estimator = gsreg_fit.best_estimator_
    return gsreg, gsreg_fit_estimator


# pprint(build_model(config3))
