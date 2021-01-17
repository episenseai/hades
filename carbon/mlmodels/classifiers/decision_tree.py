from pprint import pprint
from random import sample

from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from carbon.mlmodels.utils import (
    deliverformattedResultClf,
    deliverRoCResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultMultiClassifier,
    splitTrainTestdataset,
)

# from pprint import pprint
# from datetime import datetime
from .confgin import finalconfig1


def build(confign, model_config=None):
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
    # pprint(model_config)

    clf_gini, clf_gini_fit = gridSearchDecisionTreeClf(X_train, Y_train, config, model_config)
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
            # if len(_) > 200:
            #     fpr[i], tpr[i] = sample(zip(fpr[i], tpr[i]), 200)
        else:
            fpr[i], tpr[i], _ = roc_curve(Y_test_ovr[:], Y_score_ovr[:, i], pos_label=catClasses[0])
            roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, Y_test_ovr, Y_pred_ovr, Y_score_ovr


def gridSearchDecisionTreeClf(X, Y, config, model_config=None):
    if not model_config:
        gsClf = GridSearchCV(
            DecisionTreeClassifier(criterion="gini", random_state=100),
            param_grid={
                "min_samples_split": range(2, 32, 10),
                # "max_features": range(2, len(finalFeatureListGenerator(config)), 2),
                # "max_depth": range(2, len(finalFeatureListGenerator(config)), 2),
            },
            cv=config["data"]["cv"]["folds"],
            return_train_score=True,
        )
        gsClf_fit = gsClf.fit(X, Y)
        gsclf_results = [
            gsClf_fit.cv_results_,
            gsClf_fit.best_estimator_,
            gsClf_fit.best_score_,
            gsClf_fit.best_params_,
            gsClf_fit.best_index_,
        ]
        pprint(gsclf_results)
        gsClf_fit_estimator = gsClf_fit.best_estimator_
        # print(gsClf_fit.best_params_, gsClf_fit.best_score_)
        return gsClf, gsClf_fit_estimator
    else:
        # print(model_config)
        gsClf = GridSearchCV(
            DecisionTreeClassifier(random_state=100),
            param_grid=model_config,
            cv=config["data"]["cv"]["folds"],
            return_train_score=True,
        )
        gsClf_fit = gsClf.fit(X, Y)
        gsclf_results = [
            gsClf_fit.cv_results_,
            gsClf_fit.best_estimator_,
            gsClf_fit.best_score_,
            gsClf_fit.best_params_,
            gsClf_fit.best_index_,
        ]
        pprint(gsclf_results)
        gsClf_fit_estimator = gsClf_fit.best_estimator_
        # print(gsClf_fit.best_params_, gsClf_fit.best_score_)
        return gsClf, gsClf_fit_estimator


default_hyperparameters = {
    "criterion": ["gini"],
    # criterion:{“gini”, “entropy”}, default=”gini”
    # The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    "splitter": ["best"],  # splitter:{“best”, “random”}, default=”best”
    # The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    "max_depth": [None],  # max_depth: int, default=None
    # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    "min_samples_split": [2],  # "min_samples_split":int or float, default=2
    # The minimum number of samples required to split an internal node:
    "min_samples_leaf": [1],  # min_samples_leaf: int or float, default=1
    # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
    "min_weight_fraction_leaf": [0.0],  # min_weight_fraction_leaf: float, default=0.0
    # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    "max_features": [None],  # max_features: int, float or {“auto”, “sqrt”, “log2”}, default=None
    # The number of features to consider when looking for the best split:
    "max_leaf_nodes": [None],  # max_leaf_nodes: int, default=None
    # Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
}
model_config1 = default_hyperparameters
model_config1["criterion"] = ["gini", "entropy"]
model_config1["max_features"] = range(10, finalconfig1["data"]["data"]["includedFeatures"], 10)  # type: ignore
model_config1["min_samples_split"] = range(2, 32, 10)

# build(finalconfig1, model_config=model_config1)
