import csv
import time

beginn = time.time()

with open("./Daten/train.csv", "r") as file:
    global csvTrain
    csvTrain = [tuple(row) for row in list(csv.reader(file))]


with open("./Daten/test.csv", "r") as file:
    global csvTest
    csvTest = [tuple(row) for row in csv.reader(file)]

end = time.time()

print(str(end-beginn) + "So lange hat das gedauert" + "\n\n\n")
