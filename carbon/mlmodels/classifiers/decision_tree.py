from carbon.mlmodels.utils import (
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    loadData,
    labelEncodeCategoricalVarToNumbers,
    splitTrainTestdataset,
    deliverRoCResult,
    deliverformattedResultClf,
    metricResultMultiClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# from pprint import pprint
# from datetime import datetime
# # from Models.config import config1, config2, config4


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
    clf_gini, clf_gini_fit = gridSearchDecisionTreeClf(X_train, Y_train, config)
    # print("end", datetime.now())

    # Model Evaluation
    Y_gini_pred = clf_gini_fit.predict(X_test)
    Y_gini_score = clf_gini_fit.predict_proba(X_test)
    # print(clf_gini.best_params_, clf_gini.best_score_)
    # print(clf_gini_fit.get_depth())
    confusion = confusion_matrix(Y_test, Y_gini_pred)

    # Plot of a ROC curve for a specific class
    (
        fpr,
        tpr,
        roc_auc,
        Y_test_ovr,
        Y_pred_ovr,
        Y_score_ovr,
    ) = rocCurveforMultiClassDecisionTree(X, Y, catClasses, clf_gini_fit)
    # plotRoCCurve(catClasses, fpr, tpr, roc_auc)

    metricResult = metricResultMultiClassifier(Y_test, Y_gini_pred, Y_gini_score)
    roc = deliverRoCResult(catClasses, fpr, tpr, roc_auc)

    return (
        deliverformattedResultClf(config, catClasses, metricResult, confusion, roc),
        clf_gini_fit,
    )


def rocCurveforMultiClassDecisionTree(X, Y, catClasses, clfObject):
    # binarize the target variable
    if catClasses.shape[0] > 2:
        Y = label_binarize(Y, classes=Y.unique())
    else:
        pass
    # split the new dataset
    X_train_ovr, X_test_ovr, Y_train_ovr, Y_test_ovr = train_test_split(X, Y, random_state=0)
    clf_ovr = OneVsRestClassifier(clfObject)
    clf_ovr_fit = clf_ovr.fit(X_train_ovr, Y_train_ovr)
    Y_pred_ovr = clf_ovr_fit.predict(X_test_ovr)
    Y_score_ovr = clf_ovr_fit.predict_proba(X_test_ovr)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(Y_test_ovr.shape, Y_score_ovr.shape)
    for i in range(catClasses.shape[0]):
        if catClasses.shape[0] > 2:
            fpr[i], tpr[i], _ = roc_curve(Y_test_ovr[:, i], Y_score_ovr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], _ = roc_curve(Y_test_ovr[:], Y_score_ovr[:, i], pos_label=catClasses[0])
            roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, Y_test_ovr, Y_pred_ovr, Y_score_ovr


def gridSearchDecisionTreeClf(X, Y, config):
    gsClf = GridSearchCV(
        DecisionTreeClassifier(criterion="gini", random_state=100),
        param_grid={
            "min_samples_split": range(2, 32, 10),
            # "max_features": range(2, len(finalFeatureListGenerator(config)), 2),
            # "max_depth": range(2, len(finalFeatureListGenerator(config)), 2),
        },
        cv=config["data"]["cv"]["folds"],
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    # print(gsClf_fit.best_params_, gsClf_fit.best_score_)
    return gsClf, gsClf_fit_estimator


# pprint(build_model(config4))
