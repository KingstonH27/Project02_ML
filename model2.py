#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd
#https://chatgpt.com/share/6ace8d81-2c30-83e9-be64-d4abaa1b0d79
#https://chatgpt.com/share/6ad04906-1fac-83ea-af28-cfec5c8c5d80

import pandas as pd
import random
import matplotlib.pyplot as plt
import sympy as sp
import time


# General Variables
RESET = '\033[0m'

DATA = pd.read_csv("housing.csv")
#DATA = DATA.sort_values("median_house_value")




def getData(inputColumn,rows):
   return DATA.iloc[:rows,inputColumn].values.tolist()
def getAnswers(answerColumn,rows):
   return DATA.iloc[:rows,answerColumn].values.tolist()

#Main settings

#first column in table is column 0
answerCol = 8
inputCol = 5
rowCount = 10

DATAinputs = getData(inputCol,rowCount)
Dataanswers = getAnswers(answerCol,rowCount)

finishThreshold = 5000
maxTrainingSteps = 2000
learningRate = 0.25

optimizeLearningRate = False

inputs = DATAinputs
answers = Dataanswers

minInput = min(inputs)
maxInput = max(inputs)

notNormalizedInputs = inputs.copy()
# Normalization to -4 to 4
inputs = [
    -4 + (x - minInput) * 8 / (maxInput - minInput)
    for x in inputs
]

#Manual inputs
#inputs = [2,3,4,2]
#answers = [1,3,4,4]


def getRandomWeights():
    return [random.randint(-10, 10) / 10] + [random.randint(-10, 10) / 10 for _ in range(3)]

weights = getRandomWeights()

#Putting learning rate at 0.5 will result in exact answers when only trained on one set of inputs because it sets the bias to the target number

#Calculates model output with determined weights and inputs
def model(ins):

    y = 0
    for term in range(4):

        y += weights[term]/10**term * ins**term
    return y


b,w1,w2,w3= sp.symbols("b,w1,w2,w3")
x = sp.symbols("x")
y = sp.symbols("y")

weightSymbols = [b,w1,w2,w3]

#Mirrors the model but symbolically so that the derivates can be taken
prediction = b + w1/10 * x + w2/100* x**2 + w3/1000*x**3
loss = (y - prediction) ** 2

derivatives = []
for i in range(4):
    derivatives.append(sp.diff(loss, weightSymbols[i]))

#Derivatives taken of loss with respect to each weight to determine which direction of changing the weight decreases the loss
def updateWeights(actual,inputSet):

    weightSet = {
        y : actual,
        x : inputs[inputSet],
        b : weights[0],
        w1 : weights[1],
        w2 : weights[2],
        w3 : weights[3]
    }

    for w in range(len(weights)):
        result = derivatives[w].subs(weightSet)
        weights[w] -= result * learningRate


#Trains the data on different input and answer pairs from the dataset
def train():
    step = 0
    trainTime = time.perf_counter()

    while step < maxTrainingSteps and not all(
            abs(model(inputs[i]) - answers[i]) < finishThreshold
            for i in range(len(inputs))
    ):
        inputSet = step % len(inputs)
        updateWeights(answers[inputSet], inputSet)
        step += 1

    trainTime = time.perf_counter() - trainTime
    return step, trainTime

#Tries many different learning rates and returns the one that learns in the fewest training steps
def findBestLearningRate():
    findTimer = time.perf_counter()
    global inputs, answers, weights, learningRate
    startingInputs = inputs.copy()
    startingAnswers = answers.copy()
    startingWeights = weights.copy()
    testRateInputs = []
    testRateSteps = []

    #Find Best Learning Rate Settings
    numberToStartWith = 1
    numberOfDivisions = 6
    numberToDivideWith = 1.2
    trials = 4

    test = numberToStartWith
    for i in range(numberOfDivisions):
        testRateInputs.append(test)
        sum = 0
        for j in range(trials):
            inputs = startingInputs.copy()
            answers = startingAnswers.copy()
            weights = getRandomWeights()
            learningRate = test
            s, t = train()
            sum += s
            #print("Same rate test ", j)
        print("Average steps: ",sum / trials)
        testRateSteps.append(sum / trials)
        test /= numberToDivideWith
        print("Tested rate: ", test)

    bestVal = testRateSteps[0]
    best = 0
    for i in range(len(testRateSteps)):
        if testRateSteps[i] < bestVal:
            bestVal = testRateSteps[i]
            best = i

    print("\n**Best learning rate to minimize steps: ", testRateInputs[best])
    inputs = startingInputs.copy()
    answers = startingAnswers.copy()
    weights = startingWeights.copy()
    findTimer = time.perf_counter() - findTimer
    print("Time to find best learning rate: ",findTimer)
    return testRateInputs[best]


# ----- Main -----
if optimizeLearningRate:
    print("Finding best learning rate...")
    learningRate = findBestLearningRate()

print("\n--------------Initial Guesses----------------")
#Initial Guess
print("Learning Rate: ",learningRate)
print("Weights: ", weights)

for i in range(len(inputs)):
    print("Initial Guesses",inputs[i],": ",model(inputs[i]))


trainedMeta = train()
trainingSteps = trainedMeta[0]
trainingTime = trainedMeta[1]


print("\n--------------Trained Outputs----------------")
error = 0
totalError = 0
print(f"{'Input':<12} | {'Predicted':<15} | {'Target':<15}")
for i in range(len(inputs)):
    p = round(model(inputs[i]),5)
    a = round(answers[i],5)
    PREDICTEDCOLOR = '\033[92m' if abs(a-p)<finishThreshold else '\033[0m'
    print(f"{round(notNormalizedInputs[i],5):<12} | {PREDICTEDCOLOR}{p:<15}{RESET}| {a:<15}")

    error += round(abs(a-p),3)
    totalError += a

print("\nError: ", round(round(error,4)/round(totalError,4) * 100,2),"%")
print("Finished training in",trainingTime,"s in ",trainingSteps," steps, meeting the threshold of ",finishThreshold,".")


print("\n--------------Weights & Biases----------------")
print("*First value is bias")
print(weights)

print("\n--------------Predict new value----------------")
while True:
    v = float(input("Enter a value to predict output: "))
    normalized = -4 + (v - minInput) * 8 / (maxInput - minInput)
    print("Model output: ",model(normalized))
