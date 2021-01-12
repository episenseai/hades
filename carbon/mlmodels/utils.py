import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
)
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .config import mlmodels_config
from random import sample
import tempfile
import time
import codecs

# from pprint import pprint


def typeOfColumn(columntype):
    if columntype == object:
        return "Text"
    else:
        return "Number"


def typeOfColumn1(dfColumn):
    if dfColumn.dtype == object:
        if dfColumn.nunique() <= (int(dfColumn.shape[0]) * 0.3):
            return "Category"
        else:
            return "Text"
    else:
        return "Number"


def typeOfColumnUserUpdated(config):
    columnProperty = config["prepare:POST"]["data"]["cols"]
    columnType = {}
    for element in columnProperty:
        columnType[element["name"]] = element["type"]
    return columnType


def finaltypeOfColumnUserUpdated(finalConfig):
    columnProperty = finalConfig["data"]["cols"]
    columnType = {}
    for element in columnProperty:
        columnType[element["name"]] = element["type"]
    return columnType


def uniqueColumnIdGenerator(dfDataset):
    i = 0
    uniqueColumnId = {}
    for keys in dfDataset.columns:
        i = i + 1
        uniqueColumnId[keys] = i
    return uniqueColumnId


def uniqueColumnIdUserUpdated(config):
    columnProperty = config["prepare:POST"]["data"]["cols"]
    uniqueColumnId = {}
    for element in columnProperty:
        uniqueColumnId[element["name"]] = element["id"]
    return uniqueColumnId


def csvFileSelector(config):
    df = None
    try:
        zip_file = ZipFile(mlmodels_config.job.uploads_folder + "/"
                           + config["consume:POST"]["data"]["filepath"])
        csv_files = ""
        for csv_file in zip_file.infolist():
            if csv_file.filename.endswith(".csv"):
                if "/" in csv_file.filename or "/" in csv_file.filename:
                    pass
                else:
                    csv_files = csv_files + csv_file.filename
        # print("file:  ", csv_files)
        df = pd.read_csv(zip_file.open(csv_files))
    except UnicodeDecodeError as ex:
        # print(ex)
        with tempfile.TemporaryDirectory(dir=mlmodels_config.jobs.temp_folder) as tmpdirname:
            # print("created temporary directory: ", tmpdirname)
            zip_file.extractall(tmpdirname)
            with codecs.open(tmpdirname + "/" + csv_files, "r", encoding="utf-8", errors="ignore") as myfile:
                time.sleep(10)
                # print(myfile.read())
                df = pd.read_csv(myfile)
                # print(df)
    finally:
        return df
    # zip_file = ZipFile(
    #     mlmodels_config.jobs.uploads_folder + "/" + config["consume:POST"]["data"]["filepath"]
    # )
    # csv_files = ""
    # for csv_file in zip_file.infolist():
    #     if csv_file.filename.endswith(".csv"):
    #         if "/" in csv_file.filename or "/" in csv_file.filename:
    #             pass
    #         else:
    #             csv_files = csv_files + csv_file.filename
    # df = pd.read_csv(zip_file.open(csv_files))
    # return df


def finalCsvFileSelector(finalConfig):
    df = None
    try:
        zip_file = ZipFile(mlmodels_config.jobs.uploads_folder + "/" + finalConfig["data"]["filepath"])
        csv_files = ""
        for csv_file in zip_file.infolist():
            if csv_file.filename.endswith(".csv"):
                if "/" in csv_file.filename or "/" in csv_file.filename:
                    pass
                else:
                    csv_files = csv_files + csv_file.filename
        # print("file:  ", csv_files)
        df = pd.read_csv(zip_file.open(csv_files))
    except UnicodeDecodeError as ex:
        # print(ex)
        with tempfile.TemporaryDirectory(dir=mlmodels_config.jobs.temp_folder) as tmpdirname:
            # print("created temporary directory: ", tmpdirname)
            zip_file.extractall(tmpdirname)
            with codecs.open(tmpdirname + "/" + csv_files, "r", encoding="utf-8", errors="ignore") as myfile:
                time.sleep(10)
                # print(myfile.read())
                df = pd.read_csv(myfile)
                # print(df)
    finally:
        return df
    # zip_file = ZipFile(mlmodels_config.jobs.uploads_folder + "/" + finalConfig["data"]["filepath"])
    # csv_files = ""
    # for csv_file in zip_file.infolist():
    #     if csv_file.filename.endswith(".csv"):
    #         if "/" in csv_file.filename or "/" in csv_file.filename:
    #             pass
    #         else:
    #             csv_files = csv_files + csv_file.filename
    # # print(csv_files)
    # df = pd.read_csv(zip_file.open(csv_files))
    # return df


def binCreationCategory(dfColumn):
    # prepare the dictionary of unique values and their counts
    binDict = dfColumn.dropna().value_counts()

    totalSum1 = dfColumn.dropna().count()
    breaks = []
    counts = []
    i = 0
    sum = 0
    for key in binDict.keys():
        i = i + 1
        if i < 10:
            counts.append(int(binDict[key]))
            breaks.append(":" + str(key))
            sum = sum + binDict[key]
        elif i == 10:
            counts.append(int(totalSum1 - sum))
            breaks.append("Other bins")
        else:
            break

    # print(breaks)
    # print(counts)
    bins = {
        "breaks": breaks,
        "counts": counts,
    }
    return bins


def iterateNestedDict(dictToIterate):
    for k, v in dictToIterate.items():
        if isinstance(v, dict):
            iterateNestedDict(v)
        elif isinstance(v, list):
            iterateNestedList(v)
        else:
            if pd.isna(v):
                dictToIterate[k] = None
            else:
                continue
    return dictToIterate


def iterateNestedList(listToIterate):
    for elements in listToIterate:
        if isinstance(elements, list):
            iterateNestedList(elements)
        elif isinstance(elements, dict):
            iterateNestedDict(elements)
        else:
            if pd.isna(elements):
                i = None
            else:
                continue
    return listToIterate


def binCreation(dfColumn):
    breaks = []
    counts = []
    if typeOfColumn1(dfColumn) == "Number":
        minValue = dfColumn.dropna().min()
        maxValue = dfColumn.dropna().max()
        counter = minValue
        while counter < maxValue:
            counterLowbound = counter
            counter = round((counter + (maxValue - minValue) / 20), 4)
            # print(counterLowbound, (maxValue - counter))
            counts.append((dfColumn[dfColumn < counter]).count()
                          - (dfColumn[dfColumn < counterLowbound]).count())
            breaks.append(counter)
    else:
        pass
    result = [breaks, counts]

    return result


def correlate(dfColumn, dfTargetColumn):
    if typeOfColumn1(dfTargetColumn) == "Number":
        return float(round((dfColumn.corr(dfTargetColumn)), 2))
    else:
        return None


def featureListGenerator(config):
    df = csvFileSelector(config)
    featureList = []
    featuresPropertyList = config["transform:POST"]["data"]["features"]
    for element in featuresPropertyList:
        if element["include"]:
            featureList.append(element["name"])
    return featureList


def finalFeatureListGenerator(finalConfig):
    df = finalCsvFileSelector(finalConfig)
    featureList = []
    columnPropertyList = finalConfig["data"]["cols"]
    for element in columnPropertyList:
        if element["include"]:
            featureList.append(element["name"])
    return featureList


def featureIncludedDefault(dfColumn):
    Include = False
    if int(dfColumn.isnull().sum()) <= int(dfColumn.shape[0]) * 0.5:
        Include = True
    else:
        pass
    return Include


def loadData(config):
    dfLoaded = finalCsvFileSelector(config)
    # setting featureSet list; target and factor dataset
    featureSet = finalFeatureListGenerator(config)
    if config["data"]["target_name"] not in featureSet:
        featureSet.append(config["data"]["target_name"])
    else:
        pass

    df = dfLoaded[featureSet].dropna()
    Y = df[config["data"]["target_name"]]
    X = df.drop(config["data"]["target_name"], 1)
    return df, X, Y


def labelEncodeCategoricalVarToNumbers(X, columnType):
    for labels in X.columns:
        if columnType[labels] == "Category":
            X[labels].astype(str)
            X[labels] = LabelEncoder().fit_transform(X[labels])
        else:
            pass
    return X


def splitTrainTestdataset(X, Y, config):
    if config["data"]["sampleUsing"] == "Stratified":
        stratify = Y
    else:
        stratify = None
    X_train, X_Test, Y_train, Y_test = train_test_split(
        X,
        Y,
        random_state=100,
        test_size=(config["data"]["cv"]["holdout"]) / 100,
        stratify=stratify,
    )
    return X_train, X_Test, Y_train, Y_test


def crossValidationResults(clfObject, X, Y, config):
    score = cross_validate(clfObject, X, Y, cv=config["data"]["cv"]["folds"], scoring=["accuracy"])
    # print(sorted(sklearn.metrics.SCORERS.keys()))
    # print(score["test_accuracy"])
    return True


def plotRoCCurve(catClasses, fpr, tpr, auRoC):
    plt.figure()
    # print(catClasses, catClasses.shape)
    n_classes = catClasses.shape[0]
    # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, classes in zip(range(n_classes), catClasses):
        plt.plot(
            fpr[i],
            tpr[i],
            # color=colors,
            lw=2,
            label="ROC curve of class{0} (area = {1:0.2f}) ({2})"
            "".format(classes, auRoC[i], i),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    return True


def deliverRoCResult(catClasses, fpr, tpr, roc_auc):
    n_classes = catClasses.shape[0]
    roc = []
    for key, classes in zip(fpr, catClasses):
        roc.append({
            "name": classes,
            "x": fpr[key].tolist(),
            "y": tpr[key].tolist(),
            "area": roc_auc[key],
        })
    return roc


def deliverformattedResultClf(config, catClasses, metricResult, confusion, roc):
    returnResult = {
        "classes": list(catClasses),
        "metrics": {
            "val": metricResult[config["data"]["optimizeUsing"]]
        },
        # confusion matrix: [[TP, FP], [FN, TN]]
        "cm": confusion.tolist(),
        "roc": roc,
    }
    # pprint(returnResult)
    return returnResult


def rocCurveforClassPredictProba(X_train, X_test, Y_train, Y_test, catClasses, clfObject):
    clf_fit = clfObject
    Y_pred = clf_fit.predict(X_test)
    Y_score = clf_fit.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(Y_test.shape, Y_score.shape)
    for i in range(catClasses.shape[0]):
        if catClasses.shape[0] > 2:
            fpr[i], tpr[i], _ = roc_curve(Y_test[:], Y_score[:, i], pos_label=catClasses[i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], _ = roc_curve(Y_test[:], Y_score[:, i], pos_label=catClasses[0])
            roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, Y_pred, Y_score


def rocCurveforClassDecisionFunction(X_train, X_test, Y_train, Y_test, catClasses, clfObject):
    clf_fit = clfObject
    Y_pred = clf_fit.predict(X_test)
    Y_score = clf_fit.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(Y_test.shape, Y_score.shape)
    for i in range(catClasses.shape[0]):
        if catClasses.shape[0] > 2:
            fpr[i], tpr[i], _ = roc_curve(
                Y_test[:],
                Y_score[:, i],
                pos_label=catClasses[1],
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], _ = roc_curve(Y_test[:], Y_score[:], pos_label=catClasses[i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc, Y_pred, Y_score


def metricResultMultiClassifier(Y_test, Y_pred, Y_score):
    try:
        logloss = sklearn.metrics.log_loss(Y_test, Y_score)
    except ValueError:
        logloss = None

    try:
        precision_score = sklearn.metrics.precision_score(Y_test, Y_pred, average="macro")
    except ValueError:
        precision_score = None

    try:
        balanced_accuracy_score = sklearn.metrics.balanced_accuracy_score(Y_test, Y_pred)
    except ValueError:
        balanced_accuracy_score = None

    try:
        brier_score_loss = sklearn.metrics.brier_score_loss(Y_test, Y_score)
    except ValueError:
        brier_score_loss = None

    try:
        fbeta_score = sklearn.metrics.fbeta_score(Y_test, Y_pred, beta=0.5)
    except ValueError:
        fbeta_score = None

    try:
        hamming_loss = sklearn.metrics.hamming_loss(Y_test, Y_pred)
    except ValueError:
        hamming_loss = None

    try:
        jaccard_score = sklearn.metrics.jaccard_score(Y_test, Y_pred)
    except ValueError:
        jaccard_score = None

    try:
        matthews_corrcoef = sklearn.metrics.matthews_corrcoef(Y_test, Y_pred)
    except ValueError:
        matthews_corrcoef = None

    try:
        zero_one_loss = sklearn.metrics.zero_one_loss(Y_test, Y_pred)
    except ValueError:
        zero_one_loss = None

    try:
        f1_score = sklearn.metrics.f1_score(Y_test, Y_pred)
    except Exception as ex:
        f1_score = None

    try:
        accuracy_score = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    except ValueError:
        accuracy_score = None

    try:
        average_precision_score = sklearn.metrics.average_precision_score(Y_test, Y_score)
    except ValueError:
        average_precision_score = None

    try:
        recall_score = sklearn.metrics.recall_score(Y_test, Y_pred)
    except ValueError:
        recall_score = None

    try:
        roc_auc_score = sklearn.metrics.roc_auc_score(Y_test, Y_score)
    except Exception as ex:
        roc_auc_score = None

    metricResult = {
        "Log Loss": logloss,
        "precision score": precision_score,
        "balanced accuracy score": balanced_accuracy_score,
        "brierscore loss": brier_score_loss,
        "fbeta score": fbeta_score,
        "hamming loss": hamming_loss,
        "jaccard score": jaccard_score,
        "matthews corrcoef": matthews_corrcoef,
        "zero-one loss": zero_one_loss,
        "f1 score": f1_score,
        "accuracy": accuracy_score,
        "average precision score": average_precision_score,
        "recall score": recall_score,
        "Area under ROC curve": roc_auc_score,
    }
    # print(metricResult)
    return metricResult


def oneHotEncodeCategoricalVar(X, columnType):
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    trans_features = []
    for labels in X.columns:
        if columnType[labels] == "Category":
            X[labels].astype(str)
            trans_features.append(labels)
            # ct = ColumnTransformer([("cat", categorical_transformer, labels)])
            # X[labels] = ct.fit_transform(X[labels])
        else:
            pass
    # print(trans_features)
    ct = ColumnTransformer([("cat", categorical_transformer, trans_features)])
    X = ct.fit_transform(X)
    return X


def plotPredictedVsTrueCurve(Y_pred, Y_test, X_test, modelName):
    plt.figure()
    plt.scatter(
        X_test["annual_inc"],
        Y_test,
        s=20,
        edgecolor="black",
        c="darkorange",
        label="Actual Y",
    )
    plt.scatter(
        X_test["annual_inc"],
        Y_pred,
        s=20,
        edgecolor="black",
        c="cornflowerblue",
        label="Predicted Y",
    )
    # plt.plot(X_test["annual_inc"], Y_pred, color="cornflowerblue", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title(modelName)
    plt.legend()
    plt.show()
    return True


def metricResultRegressor(Y_test, Y_pred, Y_score):
    try:
        msle = mean_squared_log_error(Y_test, Y_pred)
    except ValueError:
        msle = None
    metricResult = {
        "explained_variance_score": explained_variance_score(Y_test, Y_pred),
        "max_error": max_error(Y_test, Y_pred),
        "mean_absolute_error": mean_absolute_error(Y_test, Y_pred),
        "mean_squared_error": mean_squared_error(Y_test, Y_pred),
        "mean_squared_log_error": msle,
        "median_absolute_error": median_absolute_error(Y_test, Y_pred),
        "r2_score": r2_score(Y_test, Y_pred),
    }
    # print(metricResult)
    return metricResult


def deliverformattedResult(config, metricResult, Y_pred, Y_test):
    # predicted vs true plot
    if Y_test.shape[0] > 400:
        predicted_values = []
        for test, pred in zip(Y_test, Y_pred):
            predicted_values.append([test, pred])
        predicted_sample = sample(predicted_values, 200)
        x1 = []
        y1 = []
        for pt in predicted_sample:
            x1.append(pt[0])
            y1.append(pt[1])
    returnResult = {
        "metrics": {
            "val": round(metricResult[config["data"]["optimizeUsing"]], 2),
            # R-Squared
            "r2": round(metricResult["r2_score"], 2),
            # RMSE
            "rmse": round(metricResult["mean_squared_error"], 2),
            # MAE : Mean Absolute Error
            "mae": round(metricResult["mean_absolute_error"], 2),
            # MAPE : Mean Squared Log Error
            "msle": metricResult["mean_squared_log_error"],
        },
        # predicted vs true plot
        "pt": {
            "p": [round(float(x), 2) for x in Y_pred],
            "t": list(Y_test)
        } if Y_test.shape[0] < 400 else {
            "p": x1,
            "t": y1
        },
    }
    return returnResult


# def binCreationCategory(dfColumn):
#    # prepare the dictionary of unique values and their counts
#    binDict = dfColumn.dropna().value_counts()
#    # reverse the dictionary to prepare the dictionary of counts and unique values
#    binspair = {}
#    for key in binDict.keys():
#        binspair[binDict[key]] = key
#    catCounts = []
#    for key in binspair.keys():
#        catCounts.append(key)
#    catCounts.sort(reverse=True)
#    totalSum = 0
#    for element in catCounts:
#        totalSum = totalSum + element
#
#    breaks = []
#    counts = []
#    i = 0
#    sum = 0
#    for element in catCounts:
#        i = i + 1
#        if i <= 10:
#            counts.append(element)
#            breaks.append(binspair[element])
#            sum = sum + element
#            if i == 10:
#                counts.append(totalSum - sum)
#                breaks.append("Other bins")
#            else:
#                pass
#        else:
#            break
#    # print(breaks)
#    # print(counts)
#    bins = {
#        "breaks": breaks,
#        "counts": counts,
#    }
#    return bins

# clfFileName = "DecisionTreeClassifier_" + str("configid") + ".joblib"
# filename must equal to result given by gridSearchDecisionTreeClf
# clfFitFileName = "DecisionTreeClassifier_Fit_" + str("configid") + ".joblib"
# my_file = Path(clfFitFileName)
# if my_file.is_file():
#    print("Using saved file")
#    clf_gini = load(clfFileName)
#    clf_gini_fit = load(clfFitFileName)
# else:
#    clf_gini, clf_gini_fit = gridSearchDecisionTreeClf(X_train, Y_train, config)

# dump(gsClf, "DecisionTreeClassifier_" + str("configid") + ".joblib")
# dump(
#    gsClf_fit_estimator,
#    "DecisionTreeClassifier_Fit_" + str("configid") + ".joblib",
# )

# def trainUsingGiniDecisionTreeClassifier(X_train, Y_train):
#    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100)
#    clf_gini_fit = clf_gini.fit(X_train, Y_train)
#    return clf_gini, clf_gini_fit
