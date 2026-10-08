#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd

import pandas as pd
import random
import matplotlib.pyplot as plt

DATA = pd.read_csv("test.csv")
DATA = DATA.sort_values("median_house_value")
startCol = 1
endCol = 6

weights = [random.randint(-10, 10)/10 for _ in range(endCol-startCol)]
bias = 0

def model(inputs):
    y = bias
    for inp in range(len(inputs)):
        y += weights[inp]*inputs[inp]
    return y

prediction = model([1,1])

def updateWeight():


print(prediction)