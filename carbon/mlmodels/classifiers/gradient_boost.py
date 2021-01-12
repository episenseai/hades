from sklearn.ensemble import GradientBoostingClassifier
from carbon.mlmodels.utils import (
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    loadData,
    labelEncodeCategoricalVarToNumbers,
    splitTrainTestdataset,
    deliverRoCResult,
    deliverformattedResultClf,
    rocCurveforClassDecisionFunction,
    metricResultMultiClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# from pprint import pprint
# from datetime import datetime
# from Models.config import config1, config2, config4


def build(confign):
    config = confign["data"]
    finalFeatureSet = finalFeatureListGenerator(config)
    columnType = finaltypeOfColumnUserUpdated(config)
    df, X, Y = loadData(config)
    Y = Y.astype(str)
    catClasses = Y.unique()
    # Encode the feature values from strings to numerical values
    X = labelEncodeCategoricalVarToNumbers(X, columnType)
    # Make the train test split, default = 75%
    X_train, X_test, Y_train, Y_test = splitTrainTestdataset(X, Y, config)

    # print("start", datetime.now())
    clf, clf_fit = gridSearchGradientBoostingClf(X_train, Y_train, config)
    # print("end", datetime.now())

    # Plot of a ROC curve for a specific class
    fpr, tpr, roc_auc, Y_pred, Y_score = rocCurveforClassDecisionFunction(X_train, X_test, Y_train, Y_test,
                                                                          catClasses, clf_fit)
    confusion = confusion_matrix(Y_test, Y_pred)
    metricResult = metricResultMultiClassifier(Y_test, Y_pred, Y_score)
    # plotRoCCurve(catClasses, fpr, tpr, roc_auc)
    roc = deliverRoCResult(catClasses, fpr, tpr, roc_auc)
    return (
        deliverformattedResultClf(config, catClasses, metricResult, confusion, roc),
        clf_fit,
    )


def gridSearchGradientBoostingClf(X, Y, config):
    gsClf = GridSearchCV(
        GradientBoostingClassifier(n_iter_no_change=5, tol=0.01, random_state=0),
        param_grid={
            "n_estimators": [50, 100],
            "max_features": ["auto", "sqrt"],
            # "max_depth": range(2, len(config["Included_features"]), 2),
            # "min_samples_split": range(2, 100, 10),
        },
        cv=config["data"]["cv"]["folds"],
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    return gsClf, gsClf_fit_estimator


# pprint(build_model(config1))
