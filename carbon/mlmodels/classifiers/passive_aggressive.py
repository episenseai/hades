from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# from pprint import pprint
# from datetime import datetime
# from Models.config import config1, config2, config4


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

    # print("start", datetime.now())
    clf, clf_fit, clf_results = gridSearchPassiveAggressiveClf(X_train, Y_train, config)
    # print("end", datetime.now())

    # Plot of a ROC curve for a specific class
    fpr, tpr, roc_auc, Y_pred, Y_score = rocCurveforClassDecisionFunction(
        X_train, X_test, Y_train, Y_test, catClasses, clf_fit
    )
    confusion = confusion_matrix(Y_test, Y_pred)
    metricResult = metricResultMultiClassifier(Y_test, Y_pred, Y_score)
    # plotRoCCurve(catClasses, fpr, tpr, roc_auc)
    roc = deliverRoCResult(catClasses, fpr, tpr, roc_auc)
    return (
        deliverformattedResultClf(config, catClasses, metricResult, confusion, roc, grid_results=clf_results),
        clf_fit,
    )


def gridSearchPassiveAggressiveClf(X, Y, config):

    steps = [
        ("scalar", StandardScaler()),
        (
            "clf",
            PassiveAggressiveClassifier(random_state=100, early_stopping=True, class_weight="balanced"),
        ),
    ]
    make_pipeline = Pipeline(steps)
    gsClf = GridSearchCV(
        make_pipeline,
        param_grid={"clf__C": [0.1, 1, 10]},
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
    return gsClf, gsClf_fit_estimator, gsclf_results


# pprint(build_model(config1))
