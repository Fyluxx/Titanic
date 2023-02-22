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

survivedCell = 1
testSize = 0.2
batch_size = 50


class Category(IntEnum):
    Survived = 1
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


class Embarked(IntEnum):
    C = 0
    Q = 1
    S = 2


class Sex(IntEnum):
    male = 0
    female = 1


def CastData(data, column):
    column = Category(column)
    match column:
        case Category.Name:
            data = zlib.crc32(data.encode()) & 0xffffffff
        case Category.Sex:
            data = Sex[data]
        case Category.Age:
            try:
                data = float(data)
            except:
                data = 0.0
        case Category.Ticket:
            data = zlib.crc32(data.encode()) & 0xffffffff
        case Category.Fare:
            data = zlib.crc32(data.encode()) & 0xffffffff
        case Category.Cabin:
            data = zlib.crc32(data.encode()) & 0xffffffff
        case Category.Embarked:
            try:
                data = Embarked[data]
            except:
                data = 0.0
    return np.float64(float(data))


def SplitIntoXandY(data):
    x, y = [], []

    arrLengthFirst = len(data)
    for i in range(arrLengthFirst):
        if i == 0:
            continue
        arrLengthSecond = len(data[i])
        tup = ()
        for j in range(1, arrLengthSecond):
            if j == survivedCell:
                value = data[i][j]
                value = CastData(value, j)
                y.append(value)
            else:
                value = data[i][j]
                value = CastData(value, j)
                tup += (value,)
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
            value = data[i][j]
            value = CastData(value, k)
            tup += (value,)
        x.append(tup)
    return x


def SplitIntoTrainAndValidation(x, y, testSize):
    x_Test, y_Test = [], []

    if (type(testSize) != float):
        raise ("Testsize falscher Datentyp")
    if testSize > 1.0 or testSize < 0:
        raise ("Testsize muss zwischen 0 und 1 sein")

    arrLength = len(x)
    testLength = int(round((arrLength * testSize)))

    random.seed(100)
    for i in range(testLength):
        index = random.randint(0, len(x) - 1)
        x_Test.append(x[index])
        y_Test.append(y[index])
        x.pop(index)
        y.pop(index)

    return np.array(x), np.array(x_Test), np.array(y), np.array(y_Test)


def TrainWithNeuralNetwork():
    import tensorflow as tf

    x_train_tf = tf.constant(x_Train)
    y_train_tf = tf.constant(y_Train)
    x_test_tf = tf.constant(x_Test)
    y_test_tf = tf.constant(y_Test)

    x_train_tf = tf.keras.utils.normalize(x_train_tf, axis=1)
    x_test_tf = tf.keras.utils.normalize(x_test_tf, axis=1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
    train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))
    test_ds = test_ds.batch(batch_size)
    regulation = tf.keras.regularizers.l2(0.01)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regulation),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=regulation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=1500, validation_data=test_ds, verbose=1)

    loss, accuracy = model.evaluate(x_Test, y_Test)
    print('Test accuracy:', accuracy)


def TrainWithXGBoost():

    global bst
    bst = XGBClassifier(n_estimators=350, max_depth=20,
                        learning_rate=0.00025, objective='binary:logistic', subsample=0.3)

    eval_set = [(x_Train, y_Train), (x_Test, y_Test)]
    bst.fit(x_Train, y_Train,
            eval_set=eval_set, verbose=True)

    y_predictions = bst.predict(x_Test)
    y_pred = [round(value) for value in y_predictions]

    rightPredicts = 0
    arrLength = len(y_pred)

    for i in range(arrLength):
        if y_pred[i] == y_Test[i]:
            rightPredicts += 1

    print("\n\n\n")
    print("Score: " + str(rightPredicts / arrLength))
    print("\n\n\n")


def PredictWithXGBoost():
    global y_pred

    y_predictions = bst.predict(x_Predict)
    y_pred = [round(value) for value in y_predictions]


def ResultToCSV(result):
    startPassengerID = 892
    arrLength = len(result)
    data = []

    for i in range(arrLength):
        data.append((startPassengerID + i, result[i]))

    print(data)

    with open('predictions', 'wb') as out:
        csv_out = csv.writer(out)
        for row in data:
            csv_out.writerow(row)


rf.GetData()
x, y = SplitIntoXandY(rf.csvTrain)
x_Train, x_Test, y_Train, y_Test = SplitIntoTrainAndValidation(x, y, testSize)

print("Beginnt lernen")
TrainWithXGBoost()

x_Predict = ConvertDataToList(rf.csvTest)
print(x_Predict)
PredictWithXGBoost()
