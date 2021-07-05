import pandas as pd

from ..utils import featureListGenerator

featureSet1 = list(
    featureListGenerator(
        pd.read_csv("/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv")
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
        pd.read_csv("/Users/rajeevranjan/Downloads/pipenv-episense/data/lending_club_loans.csv")
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
    "modelType": "multi_classifier",
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
    "modelType": "multi_classifier",
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
    "modelType": "multi_classifier",
    "holdout_%": 0.25,
    "Stratified": True,
    "nCVFolds": 6,
}
