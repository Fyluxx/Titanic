from multiprocessing import Value
from traceback import print_tb
import numpy as np
import tensorflow as tf
import math
import ReadFiles as rf
import random
import zlib

survivedCell = 1
testSize = 0.4


def CastData(data, type):
   
   
   
   
   
    return


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
                value = int(value)
                y.append(value)
            else:
                value = data[i][j]
                value = CastData(value, j)
                tup += (value,)
        x.append(tup)

    return x, y


indexes = []


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
        indexes.append(index)
        x_Test.append(x[index])
        y_Test.append(y[index])
        x.pop(index)
        y.pop(index)

    return np.array(x), np.array(x_Test), np.array(y), np.array(y_Test)


x, y = SplitIntoXandY(rf.csvTrain)
x_Train, x_Test, y_Train, y_Test = SplitIntoTrainAndValidation(x, y, testSize)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_Train, y_Train,
                    epochs=1500,
                    batch_size=50,
                    validation_data=(x_Test, y_Test),
                    verbose=1)
print(history.history['loss'])

loss, accuracy = model.evaluate(x_Test, y_Test)
print('Test accuracy:', accuracy)
