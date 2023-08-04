import csv
import time
import os


def GetData():
    csvTrain = []
    csvTest = []
    
    with open(r"C:\Projekte\MLflow\train.csv", "r") as file:
        csvTrain = [tuple(row) for row in list(csv.reader(file))]

    with open(r"C:\Projekte\MLflow\test.csv", "r") as file:
        csvTest = [tuple(row) for row in csv.reader(file)]

    return csvTrain, csvTest