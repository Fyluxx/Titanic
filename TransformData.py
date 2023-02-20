#import numpy as np
#import sklearn as sk
import ReadFiles as rf

#from sk.model_selection import train_test_split

survivedCell = 1

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

x,y = SplitIntoXandY(rf.csvTrain)

    
    
    