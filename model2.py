#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd
#https://chatgpt.com/share/6ace8d81-2c30-83e9-be64-d4abaa1b0d79


import pandas as pd
import random
import matplotlib.pyplot as plt
import sympy as sp

RESET = '\033[0m'



DATA = pd.read_csv("housing.csv")
DATA = DATA.sort_values("median_house_value")




def getData(startColumn,endColumn,rows):
   return DATA.iloc[:rows,[startColumn,endColumn]].values
def getAnswers(column,rows):
   return DATA.iloc[:rows,column].values


answerCol = 9
startCol = 4
endCol = 5
rowCount = 5
inputs = [getData(startCol,endCol,rowCount)]
answers = getAnswers(startCol,rowCount)

#inputs = [[2,3,4,5]]
#answers = [2,3,4,5]
#First input doesn't do anything, its the input compensating for weights list length(due to bias)
#Inputs = [bias,x^1,x^2,x^3]
#inputs = [[0,7,1,1]]
#answers = [28,1.2,3]

trainingSteps = 100

weights = [random.randint(-10, 10)/10] + [random.randint(-20, 20)/10 for _ in range(len(max(inputs, key=len))-1)]

#Putting learning rate at 0.5 will result in exact answers when only trained on one set of inputs because it sets the bias to the target number
learningRate = 0.00000001

def model(ins):

    y = 0
    for term in range(len(weights)):
        y += weights[term] * ins[term]**term
    return y



def updateWeight(w,actual,inputSet):
    b,w1,w2,w3= sp.symbols("b,w1,w2,w3")
    ee, x1,x2, x3 = sp.symbols("ee, x1,x2,x3")
    y = sp.symbols("y")

    weightSymbols = [b,w1,w2,w3]
    inputSymbols = [ee,x1,x2,x3]

    prediction = b + w1 * x1 + w2* x2**2 + w3*x3**3
    loss = (y - prediction) ** 2

    weightSet = {}
    weightSet[y] = actual

    for i in range(len(weightSymbols)):
        weightSet[weightSymbols[i]] = weights[i]
        weightSet[inputSymbols[i]] = inputs[inputSet][i]

    differential = sp.diff(loss, weightSymbols[w])
    result = differential.subs(weightSet)
    weights[w] -= result * learningRate






print("\n--------------Initial Guesses----------------")
#Initial Guess

for i in range(len(inputs)):
    print("Intial Guess for input set",i,": ",model(inputs[i]))

for x in range(trainingSteps):
    inputSet = x % len(inputs)
    predicted = model(inputs[inputSet])
    actual = answers[inputSet]

    for w in range(len(weights)):
        updateWeight(w,actual,inputSet)

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


#plt.scatter(x,y)

plt.show()