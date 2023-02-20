#import numpy as np
#import sklearn as sk
import ReadFiles as rf

#from sk.model_selection import train_test_split

survivedCell = 1
x, y = [], []

arrLengthFirst = len(rf.csvTrain)
for i in range(arrLengthFirst):
    if i == 0:
        continue
    arrLengthSecond = len(rf.csvTrain[i])
    tup = ()
    for j in range(arrLengthSecond):
        if j == survivedCell:
            y.append(rf.csvTrain[i][j])
        else:
            tup += (rf.csvTrain[i][j],)
    x.append(tup)

print(len(x))

print(x)
    
    
    