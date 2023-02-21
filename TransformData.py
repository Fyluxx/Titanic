from multiprocessing import Value
from traceback import print_tb
import numpy as np
import tensorflow as tf
import math
import ReadFiles as rf
import random
import zlib
from enum import IntEnum

survivedCell = 1
testSize = 0.4


class Category(IntEnum):
    Id = 0
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
    return np.float16(float(data))


def SplitIntoXandY(data):
    x, y = [], []

    arrLengthFirst = len(data)
    for i in range(arrLengthFirst):
        if i == 0:
            continue
        arrLengthSecond = len(data[i])
        tup = ()
        for j in range(arrLengthSecond):
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


x, y = SplitIntoXandY(rf.csvTrain)
x_Train, x_Test, y_Train, y_Test = SplitIntoTrainAndValidation(x, y, testSize)

x_train_tf = tf.constant(x_Train)
y_train_tf = tf.constant(y_Train)
x_test_tf = tf.constant(x_Test)
y_test_tf = tf.constant(y_Test)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))

test_ds = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5000, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(2500, activation='relu'),
    tf.keras.layers.Dense(2500, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=5000,
                    batch_size=50,
                    validation_data=test_ds,
                    verbose=1)
print(history.history['loss'])

loss, accuracy = model.evaluate(x_Test, y_Test)
print('Test accuracy:', accuracy)
