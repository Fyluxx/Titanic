#import numpy as np
#import sklearn as sk
import ReadFiles as rf

#from sk.model_selection import train_test_split

survivedCell = 1
x, y = [()], []

arrLengthFirst = len(rf.csvTrain)
for i in range(1, arrLengthFirst):
    arrLengthSecond = len(rf.csvTrain[i])
    x.append(())
    for j in range (arrLengthSecond):
        if j == survivedCell:
            y.append(rf.csvTrain[i][j])
        else:
            tup =  (rf.csvTrain[i][j],)
            k = 889
            x[i] = x[i] + tup
print(len(x))

print(x)
    
    
    