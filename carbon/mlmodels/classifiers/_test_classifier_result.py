import pandas as pd
import numpy as np
from process.build import featureListGenerator
from process.prepare import typeOfColumn1
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    KFold,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    log_loss,
    SCORERS,
)
import sklearn.metrics
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from epi import Conf
from pprint import pprint
from datetime import datetime
import pickle
from joblib import load, dump
import os.path
from pathlib import Path

featureSet1 = list(
    featureListGenerator(
        pd.read_csv(
            "/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv"
        )
    )
)
featureSet1.remove("member_id")

featureSet3 = list(
    featureListGenerator(
        pd.read_csv("/Users/rajeevranjan/Downloads/pipenv-episense/data/Heart.csv")
    )
)

featureSet2 = list(
    featureListGenerator(
        pd.read_csv(
            "/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv"
        )
    )
)
featureSet2.remove("member_id")
featureSet2.remove("grade")

config1 = {
    "Id": "aa1",
    "target": 6,
    "filepath": "/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv",
    "Included_features": featureSet1,
    "targetColumn": "term",
    "optimization_metric": "accuracy",
    "modelType": "n-classifier",
    "holdout_%": 0.2,
    "Stratified": True,
    "nCVFolds": 6,
}
config2 = {
    "Id": "aa2",
    "target": 10,
    "filepath": "/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv",
    "Included_features": featureSet2,
    "targetColumn": "sub_grade",
    "optimization_metric": "accuracy",
    "modelType": "n-classifier",
    "holdout_%": 0.25,
    "Stratified": True,
    "nCVFolds": 6,
}
config3 = {
    "Id": "aa3",
    "target": 14,
    "filepath": "/Users/rajeevranjan/Downloads/pipenv-episense/data/Heart.csv",
    "Included_features": featureSet3,
    "targetColumn": "Thal",
    "optimization_metric": "accuracy",
    "modelType": "n-classifier",
    "holdout_%": 0.25,
    "Stratified": True,
    "nCVFolds": 6,
}


def classifierModelFinalResult(config):
    returnResult = {
        "page": "models",
        "modelType": config["modelType"],  # or 'regression' 'n-classfier'
        "classes": list(catClasses),
        "models": [
            {
                "id": "112233",
                "metrics": {
                    config["optimization_metric"]: metricResult[
                        config["optimization_metric"]
                    ]
                },
                "status": 0,
                # confusion matrix: [[TP, FP], [FN, TN]]
                "cm": confusion.tolist(),
                "roc": roc,
            }
        ],
    }
