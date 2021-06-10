from devtools import debug
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from carbon.mlmodels.utils import (
    convert_cvresults_tolist,
    deliverformattedResultClf,
    deliverRoCResult,
    finalFeatureListGenerator,
    finaltypeOfColumnUserUpdated,
    labelEncodeCategoricalVarToNumbers,
    loadData,
    metricResultMultiClassifier,
    splitTrainTestdataset,
)

from .confgin import finalconfig1


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

    possible_param_grid, default_hp_grid = paramlist(confign)
    model_config = confign["hyper_params"]
    if not model_config:
        model_config = default_hp_grid

    clf_gini_fit, clf_results = gridSearchDecisionTreeClf(X_train, Y_train, config, model_config)

    if not confign["hp_results"]:
        confign["hp_results"] = [
            {
                "hp_grid": model_config,
                "result": clf_results,
            },
        ]
    else:
        hp_result = {
            "hp_grid": model_config,
            "result": clf_results,
        }
        confign["hp_results"].append(hp_result)

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
    # debug(clf_results)
    return (
        deliverformattedResultClf(
            config,
            catClasses,
            metricResult,
            confusion,
            roc,
            grid_results=clf_results,
            hp_results=confign["hp_results"],
            possible_model_params=possible_param_grid,
        ),
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

    fpr = {}
    tpr = {}
    roc_auc = {}
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
    gsClf = GridSearchCV(
        DecisionTreeClassifier(random_state=100),
        param_grid=model_config,
        cv=config["data"]["cv"]["folds"],
        return_train_score=False,
    )
    gsClf_fit = gsClf.fit(X, Y)
    gsclf_results = {
        "cvresult_list": convert_cvresults_tolist(gsClf_fit.cv_results_),
        "mean_test_score": gsClf_fit.cv_results_["mean_test_score"].tolist(),
        "params": gsClf_fit.cv_results_["params"],
        # gsClf_fit.best_estimator_,
        "best_score": round(gsClf_fit.best_score_, 2),
        "best_params": list(zip(gsClf_fit.best_params_.keys(), gsClf_fit.best_params_.values())),
        # "possible_model_params": possible_model_params
        # "scorer_function": str(gsClf_fit.scorer_),
        # gsClf_fit.best_index_,
    }

    # debug(gsclf_results)
    gsClf_fit_estimator = gsClf_fit.best_estimator_
    # print(gsClf_fit.best_params_, gsClf_fit.best_score_)
    return gsClf_fit_estimator, gsclf_results


def paramlist(confign):
    config = confign["data"]
    possible_param_grid = {
        "criterion": {"default": "gini", "possible_str": ["gini", "entropy"]},  # [min,max]
        "splitter": {"default": "best", "possible_str": ["best", "random"]},
        "max_depth": {"default": None, "possible_int": [1, config["data"]["rows"]]},
        "min_samples_split": {"default": 2, "possible_int": [2, config["data"]["rows"]]},
        "min_samples_leaf": {"default": 1, "possible_int": [1, config["data"]["rows"]]},
        "max_features": {"default": None, "possible_int": [1, config["data"]["includedFeatures"]]},
        "max_leaf_nodes": {"default": None, "possible_int": [1, config["data"]["rows"]]},
    }

    default_hp_grid = {
        "min_samples_split": [2, 12, 22],  # range(2, 32, 10),
        "max_features": [2, 12],  # range(2, len(finalFeatureListGenerator(config)), 10),
    }
    return (possible_param_grid, default_hp_grid)
