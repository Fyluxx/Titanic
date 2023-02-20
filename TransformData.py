import numpy as np
import tensorflow as tf
import math
#import sklearn as sk
import ReadFiles as rf
import random

#from sk.model_selection import train_test_split

survivedCell = 1
testSize = 0.2


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
                try:
                    data = data[i][j]
                except:
                    print(data)
                print(data)
                y.append(data)
            else:
                data = data[i][j]
                if(type(data) != int):
                    data = hash(data)
                tup += (data,)
        x.append(tup)

    return x, y


def SplitIntoTrainAndValidation(x, y, testSize):
    x_Test, y_Test = [], []

    if(type(testSize) != float):
        raise("Testsize falscher Datentyp")
    if testSize > 1.0 or testSize < 0:
        raise("Testsize muss zwischen 0 und 1 sein")

    arrLength = len(x)
    testLength = int(round((arrLength * testSize)))

    for i in range(testLength):
        index = random.randint(0, len(x) - 1)
        x_Test.append(x[index])
        y_Test.append(y[index])
        x.pop(index)
        y.pop(index)

    return x, x_Test, y, y_Test


x, y = SplitIntoXandY(rf.csvTrain)
x_Train, x_Test, y_Train, y_Test = SplitIntoTrainAndValidation(x, y, testSize)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(len(x_Train[0]))),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_Train, y_Train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_Test, y_Test))

loss, accuracy = model.evaluate(x_Test, y_Test)
print('Test accuracy:', accuracy)
