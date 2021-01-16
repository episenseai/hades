from carbon.mlpipeline.utils import csvFileSelector, featureListGenerator


def process(config):
    df = csvFileSelector(config)
    # uniqueColumnId = uniqueColumnIdGenerator(df)
    target = config["transform:POST"]["data"]["target"]
    featuresPropertyList = config["transform:POST"]["data"]["features"]
    modelType = ""
    for element in featuresPropertyList:
        if element["id"] == target:
            TargetColumn = element["name"]
            if element["type"] == "Number":
                modelType = modelType + "regressor"
            elif element["type"] == "Category":
                if df[TargetColumn].nunique() <= 2:
                    modelType = modelType + "classifier"
                else:
                    modelType = modelType + "multi_classifier"
    # print(modelType)
    if modelType == "regressor":
        metrics = [
            "explained_variance_score",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        ]
    else:
        metrics = [
            "Log Loss",
            "precision score",
            "average precision score",
            "balanced accuracy score",
            "brierscore loss",
            "fbeta score",
            "hamming loss",
            "jaccard score",
            "matthews corrcoef",
            "zero-one loss",
            "f1 score",
            "accuracy",
            "recall score",
            "Area under ROC curve",
        ]

    featureList = featureListGenerator(config)
    # print(len(featureList))

    returnResult = {
        "stage": "build:GET",
        "data": {
            "summary": {
                "features": len(featuresPropertyList),
                "includedFeatures": len(featureList),
                "rows": config["transform:POST"]["data"]["rows"],
                "target": TargetColumn,
                "modelType": modelType,  # or 'Binary Calssification' or 'Regression'
            },
            "samplingMethods": ["Random"] if modelType == "regressor" else ["Random", "Stratified"
                                                                           ],  # Random - Monte Carlo Sampling
            "splitMethods": ["Cross Validation", "Training Validation Holdout"],
            "cv": {
                "folds": 5,
                "holdout": 20
            },  # between 2-20  ## between 5%-50%
            ## Data Split
            "sampleUsing": "Random",
            "splitUsing": "Cross Validation",
            "metrics": metrics,
            "optimizeUsing": "Log Loss",
            "downsampling": False,
        },
    }
    # print(returnResult)
    return returnResult


# print(transform_func(config))
