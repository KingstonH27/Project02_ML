#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd
#https://chatgpt.com/share/6ace8d81-2c30-83e9-be64-d4abaa1b0d79


import pandas as pd
import random
import matplotlib.pyplot as plt
import sympy as sp

RESET = '\033[0m'

DATA = pd.read_csv("test.csv")
DATA = DATA.sort_values("median_house_value")
startCol = 1
endCol = 2

inputs = [[0.2,0],[1,0],[1,1]]
answers = [10,1.2,3]

weights = [random.randint(-10, 10)/10 for _ in range(len(max(inputs, key=len)))]


biases = [random.randint(-10, 10)/10 for _ in range(len(inputs))]


def model(ins,):
    y = 0
    for inp in range(len(ins)):
        y += biases[inp]
        y += weights[inp]*ins[inp]
    return y




#weight= sp.symbols("weight")
#x, y = sp.symbols("x y")

#loss = (y-x)**2
#differential = sp.diff(loss, weight)

#for i in range(len(weightSet)):
#    differential.subs({
#        x : predicted,
#        y : actual,
#        weight : weights[weightSet][i]
#    })



#weightSet = weights[]
#dw1.subs({
#    b:
#})

def updateWeight(w,actual,predicted,inputSet):


    derivativeLoss = 2*(actual-predicted)

    weights[w] = weights[w] + derivativeLoss * inputs[inputSet][w] * 0.1
    biases[w] = biases[w] + derivativeLoss * 0.1

    #2(1 - -0.4)
    #2.8
    #0.5 + 0.14 = 0.64 new weight
    #1*0.64 = 0.64 new ouput
    #original output -0.4
    #d/dw l()

print("\n--------------Initial Guesses----------------")
#Initial Guess
for i in range(len(inputs)):
    print("Intial Guess for input set",i,": ",model(inputs[i]))

for x in range(200):
    inputSet = x % len(inputs)
    predicted = model(inputs[inputSet])
    actual = answers[inputSet]

    for w in range(len(weights)):
        updateWeight(w,actual,predicted,inputSet)

print("\n--------------Trained Outputs----------------")
error = 0
totalError = 0
print(f"{'Input Set':<12} | {'Predicted':<15} | {'Target':<15}")
for i in range(len(inputs)):
    p = model(inputs[i])
    a = answers[i]
    PREDICTEDCOLOR = '\033[92m' if abs(a-p)<0.5 else '\033[0m'
    print(f"{i:<12} | {PREDICTEDCOLOR}{p:<15}{RESET}| {a:<15}")

    error += abs(a-p)
    totalError += a

print("\nError: ", round(error/totalError * 100,2),"%")


print("\n--------------Weights & Biases----------------")
print(weights)
print(biases)

#plt.scatter(x,y)

plt.show()