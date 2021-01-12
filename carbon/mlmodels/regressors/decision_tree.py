from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from carbon.mlmodels.utils import (
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    loadData,
    labelEncodeCategoricalVarToNumbers,
    splitTrainTestdataset,
    metricResultRegressor,
    deliverformattedResult,
)
from sklearn.model_selection import GridSearchCV

# from Models.config import config1, config2, config3
# from datetime import datetime
# from pprint import pprint


def build(confign):
    modelName = "Decision Tree Regressor"
    config = confign["data"]
    finalFeatureSet = finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)

    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    # print("start", datetime.now())
    reg, reg_fit = gridSearchDecisionTreeRegressor(X_train, Y_train, config)
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


def gridSearchDecisionTreeRegressor(X, Y, config):
    steps = [
        ("feature_selection", SelectFromModel(LassoCV(), "median")),
        ("reg", DecisionTreeRegressor(random_state=100)),
    ]
    make_pipeline = Pipeline(steps)
    gsreg = GridSearchCV(
        make_pipeline,
        param_grid={
            "reg__min_samples_split": range(2, 32, 10),
            # "reg__penalty": ["l1", "l2"],
            # "reg__max_depth": range(2, len(finalFeatureListGenerator(config)), 2),
        },
        cv=config["data"]["cv"]["folds"],
    )
    gsreg_fit = gsreg.fit(X, Y)
    gsreg_fit_estimator = gsreg_fit.best_estimator_

    return gsreg, gsreg_fit_estimator


# pprint(build_model(config3))
