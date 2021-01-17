from carbon.mlpipeline.utils import (
    binCreation,
    binCreationCategory,
    correlate,
    csvFileSelector,
    featureIncludedDefault,
    iterateNestedDict,
    typeOfColumnUserUpdated,
    uniqueColumnIdUserUpdated,
)


def process(config):
    df = csvFileSelector(config)

    uniqueColumnId = uniqueColumnIdUserUpdated(config)
    columnType = typeOfColumnUserUpdated(config)

    target = config["prepare:POST"]["data"]["target"]
    for cols in df.columns:
        if uniqueColumnId[cols] == target:
            TargetColumn = cols

    columnPropertyList = config["prepare:POST"]["data"]["cols"]
    features = []
    for element in columnPropertyList:
        cols = element["name"]
        if columnType[cols] == "Category":
            features.append(
                {
                    "id": int(element["id"]),
                    "name": element["name"],
                    "weight": 1,
                    "include": featureIncludedDefault(df[cols]),
                    "origin": "native",
                    "type": element["type"],
                    "stats": {
                        "unique": int(df[cols].dropna().nunique()),
                        "bins": binCreationCategory(df[cols]),
                        "missing": int(df[cols].isnull().sum()),
                    },
                }
            )
        elif columnType[cols] == "Number":
            features.append(
                {
                    "id": int(element["id"]),
                    "name": element["name"],
                    "weight": 1,
                    "include": featureIncludedDefault(df[cols]),
                    "origin": "native",
                    "type": element["type"],
                    "stats": {
                        "mean": float(round((df[cols].mean(skipna=True)), 2)),
                        "median": float(round((df[cols].median(skipna=True)), 2)),
                        "std": float(round((df[cols].std(skipna=True)), 2)),
                        "min": float(round((df[cols].min(skipna=True)), 2)),
                        "max": float(round((df[cols].max(skipna=True)), 2)),
                        "missing": int(df[cols].isnull().sum()),
                        "quantile": [
                            float(round(x, 2))
                            for x in df[cols]
                            .quantile(
                                [
                                    0.05,
                                    0.10,
                                    0.15,
                                    0.20,
                                    0.25,
                                    0.30,
                                    0.35,
                                    0.40,
                                    0.45,
                                    0.50,
                                    0.55,
                                    0.60,
                                    0.65,
                                    0.70,
                                    0.75,
                                    0.80,
                                    0.85,
                                    0.90,
                                    0.95,
                                    1.00,
                                ]
                            )
                            .dropna()
                        ],
                        "bins": {
                            "breaks": [float(x) for x in binCreation(df[cols])[0]],
                            "counts": [int(x) for x in binCreation(df[cols])[1]],
                        },
                        "correlation": correlate(df[cols], df[TargetColumn]),
                    },
                }
            )
        else:
            continue

    returnResult = {
        "stage": "transform:GET",
        "data": {
            "rows": int(df.shape[0]),
            "target": target,
            ## id: unique id assigned to the feature
            ## weight: between 0 to 1, of the importance of the feature in prediction
            ## include: whether to include that feature to train the model
            ## origin: one of ['native', 'derived']
            ##   'native': comes from columns in the file uploaded
            ##   'derived': comes from a combination of other native features through transformation
            ## 20-quantiles also called vigintiles: Q1, Q2, ..., Q19
            ## Q-5(25%), Q-10 (50% = median), Q-15(75%)
            ## bins - histogram of frequency distribution
            ## correlation: correlation with the target variable
            "features": features,
        },
    }
    returnResult = iterateNestedDict(returnResult)
    # print(returnResult)
    return returnResult


# pprint(prepare_func(config))
