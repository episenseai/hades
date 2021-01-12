from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    clf, clf_fit = gridSearchLinearSVClf(X_train, Y_train, config)
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


def gridSearchLinearSVClf(X, Y, config):

    steps = [
        ("scalar", StandardScaler()),
        ("transformerNystroem", Nystroem(gamma=0.2, random_state=1)),
        ("clf", LinearSVC(random_state=100, class_weight="balanced", max_iter=1000)),
    ]
    make_pipeline = Pipeline(steps)
    gsClf = GridSearchCV(
        make_pipeline,
        param_grid={
            "clf__C": [0.1, 1, 10],
            # "clf__penalty": ["l1", "l2"],
            "clf__loss": ["hinge", "squared_hinge"],
        },
        cv=config["data"]["cv"]["folds"],
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    return gsClf, gsClf_fit_estimator


# pprint(build_model(config1))
