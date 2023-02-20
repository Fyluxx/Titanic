import numpy as np
import math
import sklearn as sk
import ReadFiles as rf

#from sk.model_selection import train_test_split

survivedCell = 1
testSize = 0.3

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
                y.append(data[i][j])
            else:
                tup += (data[i][j],)
        x.append(tup)

    return x,y

def SplitIntoTrainAndValidation(x,y, testSize):
    x_Test, y_Test = [],[]
    
    if(type(testSize) != float):
        raise("Testsize falscher Datentyp")
    if testSize > 1.0 or testSize < 0:
        raise("Testsize muss zwischen 0 und 1 sein")
     
    arrLength = len(x)    
    testLength = int(math.round((arrLength * testSize))
    
    for i in range(testLength):
        index = np.random(len(x))
        x_Test.append(x[index])
        y_Test.append(y[index])
        x.pop(index)
        y.pop(index)
    
    return x, x_Test, y, y_Test

x,y = SplitIntoXandY(rf.csvTrain)
x_Train, x_Test, y_Train, y_Test = SplitIntoTrainAndValidation(x[0:4],y[0:4], testSize)

    
    
    