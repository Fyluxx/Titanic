from multiprocessing import Value
from pickle import TRUE
from traceback import print_tb
import numpy as np
import math
import ReadFiles as rf
import random
import zlib
from xgboost import XGBClassifier
from enum import IntEnum
import csv
import mlflow
class Category(IntEnum):
    Pclass = 2
    Name = 3
    Sex = 4
    Age = 5
    SibSp = 6
    Parch = 7
    Ticket = 8
    Fare = 9
    Cabin = 10
    Embarked = 11

def CastData(data, column):
    column = Category(column)
    match column:
        case Category.Pclass:
            match int(data):
                case 1:
                    return (
                        1,
                        0,
                        0,
                    )
                case 2:
                    return (
                        0,
                        1,
                        0,
                    )
                case 3:
                    return (
                        0,
                        0,
                        1,
                    )
        case Category.Name:
            data = data.lower()

            if "mr." in data:
                return (
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif "major." in data or "col." in data or "capt." in data:
                return (
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif "rev." in data:
                return (
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif "sir." in data or "don." in data or "jonkheer." in data:
                return (
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif "dr." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif "master." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                )
            elif "mrs." in data or "mme." in data or "lady." in data or "dona." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                )
            elif "miss." in data or "mlle." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                )
            elif "ms." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                )
            elif "countess." in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                )
        case Category.Sex:
            if data == "male":
                return (
                    1,
                    0,
                )
            else:
                return (
                    0,
                    1,
                )
        case Category.Age:
            if data == "":
                return (None, 0, 0, 0, 0, 0)
            else:
                if float(data) < 18:
                    return (float(data), 1, 0, 0, 0, 1)
                elif float(data) > 50:
                    return (float(data), 0, 1, 0, 0, 1)
                elif float(data) < 30:
                    return (float(data), 0, 0, 1, 0, 1)
                else:
                    return (float(data), 0, 0, 0, 1, 1)

        case Category.Ticket:
            return (None,)
        case Category.Fare:
            if data == "":
                return (
                    None,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            elif float(data) < 10:
                return (
                    float(data),
                    1,
                    0,
                    0,
                    0,
                    0,
                )
            elif float(data) < 25:
                return (
                    float(data),
                    0,
                    1,
                    0,
                    0,
                    0,
                )
            elif float(data) < 50:
                return (
                    float(data),
                    0,
                    0,
                    1,
                    0,
                    0,
                )
            elif float(data) < 100:
                return (
                    float(data),
                    0,
                    0,
                    0,
                    1,
                    0,
                )
            else:
                return (
                    float(data),
                    0,
                    0,
                    0,
                    0,
                    1,
                )
        case Category.Cabin:
            data = data.lower()
            if "a" in data:
                return (
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                )
            elif "b" in data:
                return (
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                )
            elif "c" in data:
                return (
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                )
            elif "d" in data:
                return (
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                )
            elif "e" in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                )
            elif "f" in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                )
            elif "g" in data:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                )
            else:
                return (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

        case Category.Embarked:
            if data == "S":
                return (
                    1,
                    0,
                    0,
                    1,
                )
            elif data == "Q":
                return (
                    0,
                    1,
                    0,
                    1,
                )
            elif data == "C":
                return (
                    0,
                    0,
                    1,
                    1,
                )
            else:
                return (
                    0,
                    0,
                    0,
                    0,
                )
        case Category.SibSp:
            data = int(data)

            match data:
                case 0:
                    return (
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                case 1:
                    return (
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                case 2:
                    return (
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                    )
                case 3:
                    return (
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                    )
                case 4:
                    return (
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                    )
                case 5:
                    return (
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                    )
                case 8:
                    return (
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                    )
    try:
        x = np.float64(float(data))
        return (x,)
    except:
        print(data)
        k = 546
        raise

        print(data)
        k = 546
        raise
        print(data)
        k = 546
        raise


def SplitIntoXandY(dict_params,data):
    x, y = [], []
    arrLengthFirst = len(data)
    for i in range(arrLengthFirst):
        if i == 0:
            continue
        arrLengthSecond = len(data[i])
        tup = ()
        for j in range(1, arrLengthSecond):
            if j == dict_params["survivedCell"]:
                y.append(float(data[i][j]))
            else:
                tup += CastData(data[i][j], j)
        x.append(tup)
    return x, y


def ConvertDataToList(data):
    x = []
    arrLengthFirst = len(data)
    for i in range(arrLengthFirst):
        if i == 0:
            continue
        arrLengthSecond = len(data[i])
        tup = ()
        for j in range(1, arrLengthSecond):
            k = j + 1

            tup += CastData(data[i][j], k)
        x.append(tup)
    return x


def SplitIntoTrainAndValidation(dict_params, x, y):
    x_Test, y_Test = [], []
    arrLength = len(x)
    testLength = int(round((arrLength * dict_params["testSize"])))
    random.seed(100)
    for i in range(testLength):
        index = random.randint(0, len(x) - 1)
        x_Test.append(x[index])
        y_Test.append(y[index])
        x.pop(index)
        y.pop(index)
    return np.array(x), np.array(x_Test), np.array(y), np.array(y_Test)


def CrossValidation(dict_params,xgb_parameters, data):
    bucketsX = []
    bucketsY = []
    x_Test, x_Train, y_Train, y_Test = [], [], [], []
    random.seed(50)
    arrLength = len(data) // dict_params["crossValidation"]
    remainder = (len(data) - 1) % dict_params["crossValidation"]
    for i in range(dict_params["crossValidation"]):
        bucketsX.append([])
        bucketsY.append([])
        for j in range(arrLength):
            arrLengthSecond = len(data[1])
            tup = ()
            r = random.randint(0, len(data) - 1)
            while r == 0:
                r = random.randint(0, len(data) - 1)
            for k in range(1, arrLengthSecond):
                if k == dict_params["survivedCell"]:
                    bucketsY[i].append(float(data[r][k]))
                else:
                    tup += CastData(data[r][k], k)
            bucketsX[i].append(tup)
            data.pop(r)
    for i in range(remainder):
        arrLengthSecond = len(data[0])
        tup = ()
        r = random.randint(0, len(data) - 1)
        while r == 0:
            r = random.randint(0, len(data) - 1)
        for k in range(1, arrLengthSecond):
            if k == dict_params["survivedCell"]:
                bucketsY[i].append(float(data[r][k]))
            else:
                tup += CastData(data[r][k], k)
        bucketsX[i].append(tup)
        data.pop(r)
    sum = 0.0
    bestValue, currentValue = 0, 0
    for i in range(dict_params["crossValidation"]):
        dict_params["durchgang"] = i
        x_Train.clear()
        y_Train.clear()
        for j in range(dict_params["crossValidation"]):
            if i == j:
                x_Test = bucketsX[j]
                y_Test = bucketsY[j]
            else:
                x_Train.extend(bucketsX[j])
                y_Train.extend(bucketsY[j])
        print("Durchgang: " + str(dict_params["durchgang"] + 1))
        currentValue = TrainWithXGBoost(dict_params, xgb_parameters,x_Train, x_Test, y_Train, y_Test)
        sum += currentValue
        if currentValue > bestValue:
            bestValue = currentValue
            bestModel = i
    print("Average Score: " + str(sum / dict_params["crossValidation"])),
    for i in range(dict_params["crossValidation"]):
        dict_params["result"].append(PredictWithXGBoost(dict_params,i))
    res = AveragePredict(dict_params)
    ResultToCSV(res)

def TrainWithXGBoost(dict_params,xgb_parameters,x_Train, x_Test, y_Train, y_Test):
    bst = dict_params["bst"]
   
    bst.append(
        XGBClassifier(
            n_estimators=xgb_parameters["n_estimators"],
            max_depth=xgb_parameters["max_depth"],
            min_child_weight=xgb_parameters["min_child_weight"],
            learning_rate=xgb_parameters["learning_rate"],
            objective=xgb_parameters["obejctive"],
            subsample=xgb_parameters["subsample"],
            random_state=xgb_parameters["random_state"],
            early_stopping_rounds=xgb_parameters["early_stopping_rounds"],
        )
    )
    eval_set = [(x_Train, y_Train), (x_Test, y_Test)]
    bst[len(bst) - 1].fit(x_Train, y_Train, eval_set=eval_set, verbose=False)
    y_predictions = bst[len(bst) - 1].predict(x_Test)
    y_pred = [round(value) for value in y_predictions]
    rightPredicts = 0
    arrLength = len(y_pred)
    dict_params["bst"] = bst
    for i in range(arrLength):
        if y_pred[i] == y_Test[i]:
            rightPredicts += 1
    print("Score: " + str(rightPredicts / arrLength))
    mlflow.log_metric(
        "Durchgang " + str(dict_params["durchgang"] + 1) + " Score ", rightPredicts / arrLength
    )
    return rightPredicts / arrLength


def PredictWithXGBoost(dict_params, model_number):
    y_predictions = dict_params["bst"][model_number].predict(x_Predict)
    y_pred = [round(value) for value in y_predictions]
    return y_pred


def AveragePredict(dict_params):
    res = []
    result = dict_params["result"]
    for i in range(len(result[0])):
        count0, count1 = 0, 0
        for j in range(len(result)):
            if result[j][i] == 0:
                count0 += 1
            else:
                count1 += 1
        if count0 > count1:
            res.append(0)
        else:
            res.append(1)
    return res


def ResultToCSV(dict_params):
    result = dict_params["result"]
    startPassengerID = 892
    arrLength = len(result)
    data = []
    data.append(("PassengerId", "Survived"))
    for i in range(arrLength):
        data.append((startPassengerID + i, str(result[i])))
    itemLength = len(data[0])
    with open("predictions.csv", "w") as out:
        csv_out = csv.writer(out, delimiter=",")
        for row in data:
            st = row
            csv_out.writerow(st)


# Starte ein MLFlow Experiment

dict_params = {
    "survivedCell" : 1,
    "testSize" : 0.2,
    "crossValidation" : 5,
    "bst" : [],
    "result" : [],
    "bestModel" : 0,
    "durchgang" : 0
}

xgb_parameters = {
    "n_estimators": 10000,
    "max_depth": 9,
    "min_child_weight": 11,
    "learning_rate": 0.015,
    "obejctive": "binary:logistic",
    "subsample": 0.25,
    "random_state": 42,
    "early_stopping_rounds": 8
}

mlflow.set_experiment("Titanic")
experiment_id = mlflow.get_experiment_by_name("Titanic").experiment_id
# AWS Server URI
#remote_server_uri = "http://ec2-13-51-64-92.eu-north-1.compute.amazonaws.com:5000/"
#mlflow.set_tracking_uri(remote_server_uri)

#mlflow.run(remote_server_uri)

with mlflow.start_run(experiment_id=experiment_id, run_name="V1"):
    """mlflow.log_param("n_estimators", parameters["n_estimators"])
    mlflow.log_param("max_depth", parameters["max_depth"])
    mlflow.log_param("min_child_weight", parameters["min_child_weight"])
    mlflow.log_param("learning_rate", parameters["learning_rate"])
    mlflow.log_param("objective", parameters["obejctive"])
    mlflow.log_param("subsample", parameters["subsample"])
    mlflow.log_param("random_state", parameters["random_state"])
    mlflow.log_param("early_stopping_rounds", parameters["early_stopping_rounds"])
    mlflow.log_param("testSize", dict_params["testSize"])
    mlflow.log_param("crossValidation", crossValidation)"""
    mlflow.xgboost.autolog()
    csv_train, csv_test = rf.GetData()
    x_Predict = ConvertDataToList(csv_test)
    CrossValidation(dict_params, xgb_parameters,csvTrain)
    x, y = SplitIntoXandY(dict_params,csv_train)
    # ResultToCSV(dict_params["result"])
    mlflow.xgboost.log_model(bst[dict_params["bestModel"]], "model")
    mlflow.log_metric("Bestes Model", dict_params["bestModel"])
    mlflow.log_artifact("predictions.csv")
    mlflow.end_run()