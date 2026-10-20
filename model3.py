#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd
#https://chatgpt.com/share/6ace8d81-2c30-83e9-be64-d4abaa1b0d79
#https://chatgpt.com/share/6ad04906-1fac-83ea-af28-cfec5c8c5d80

import pandas as pd
import random
import matplotlib.pyplot as plt
import sympy as sp
import time
import ast
import os
import torch
from settings import *

# General Variables
RESET = '\033[0m'

DATA = pd.read_csv(FileName)


if not os.path.exists("weights.txt"):
    with open("weights.txt", "w") as file:
        file.write("[]")

def loadWeights():
    with open("weights.txt", "r") as file:
        content = file.read()
        return ast.literal_eval(content) if content.strip() else []

def saveWeights(w):
    for i in range(len(w)):
        w[i] = round(w[i],8)
    with open("weights.txt", "w") as file:
        file.write(str(w))
def getData(inputColumn, rows):
    return pd.to_numeric(
        DATA.iloc[:rows, inputColumn], errors="raise"
    ).tolist()

def getAnswers(answerColumn, rows):
    return pd.to_numeric(
        DATA.iloc[:rows, answerColumn], errors="raise"
    ).tolist()
#Main settings

#first column in table is column 0


DATAinputs = getData(inputCol,rowCount)
Dataanswers = getAnswers(answerCol,rowCount)



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
    return [random.randint(-10, 10) / 10 for _ in range(termCount)]


weights = getRandomWeights()

#Putting learning rate at 0.5 will result in exact answers when only trained on one set of inputs because it sets the bias to the target number

#Calculates model output with determined weights and inputs
def model(ins):
    y = 0
    for term in range(len(weights)):
        y += weights[term] / termDivisor**term * ins**term
    return y

'''
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
'''
#Derivatives taken of loss with respect to each weight to determine which direction of changing the weight decreases the loss

def updateWeights(actual, inputSet):
    ins = inputs[inputSet]
    error = actual - model(ins)

    for w in range(len(weights)):
        result = -2 * error * ins**w / termDivisor**w
        weights[w] -= result * learningRate

'''
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

'''
#Trains the data on different input and answer pairs from the dataset
def train():
    step = 0
    trainTime = time.perf_counter()
    if not optimizeLearningRate:
        print("\nTraining...")
    while step < maxTrainingSteps and not all(
            abs(model(inputs[i]) - answers[i]) < finishThreshold
            for i in range(len(inputs))
    ):
        inputSet = step % len(inputs)
        updateWeights(answers[inputSet], inputSet)
        step += 1

        if not optimizeLearningRate:
            if step%(maxTrainingSteps/progressBarWidth) == 0:
                print("", end="\r")
                print("[", end="")
                for i in range(int(step/(maxTrainingSteps/progressBarWidth))):
                    print("|", end="")
                for i in range(int((maxTrainingSteps - step)/(maxTrainingSteps/progressBarWidth))):
                    print(" ", end="")
                print("]", end="",flush=True)
    if not optimizeLearningRate:
        print("\nComplete!")

    trainTime = time.perf_counter() - trainTime
    return step, trainTime

'''
def findBestLearningRate():
    findTimer = time.perf_counter()
    global learningRate

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Move inputs and answers to the GPU
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    targets = torch.tensor(answers, dtype=torch.float32, device=device)

    # Precalculate polynomial features
    features = torch.stack([
        torch.ones_like(x),
        x / 10,
        x**2 / 100,
        x**3 / 1000
    ], dim=1)

    testRateInputs = []
    testRateSteps = []
    test = numberToStartWith

    for i in range(numberOfDivisions):
        testRateInputs.append(test)

        # Create independent weights for each trial
        w = torch.tensor(
            [getRandomWeights() for _ in range(trials)],
            dtype=torch.float32,
            device=device
        )

        finished = torch.zeros(
            trials, dtype=torch.bool, device=device
        )

        stepCounts = torch.full(
            (trials,), maxTrainingSteps,
            dtype=torch.int64, device=device
        )

        for step in range(maxTrainingSteps + 1):

            # Check whether each trial meets the original threshold
            predictions = w @ features.T

            converged = (
                torch.abs(predictions - targets.unsqueeze(0))
                < finishThreshold
            ).all(dim=1)

            newlyFinished = (~finished) & converged
            stepCounts[newlyFinished] = step
            finished |= converged

            # Stop when all trials converge or max steps are reached
            if finished.all().item() or step == maxTrainingSteps:
                break

            # Train on the same input sequence as the CPU version
            inputSet = step % len(inputs)

            prediction = w @ features[inputSet]
            error = targets[inputSet] - prediction

            gradient = (
                -2 * error.unsqueeze(1)
                * features[inputSet].unsqueeze(0)
            )

            updatedWeights = w - test * gradient

            # Preserve weights for trials that have converged
            w = torch.where(
                finished.unsqueeze(1),
                w,
                updatedWeights
            )

        averageSteps = stepCounts.float().mean().item()
        testRateSteps.append(averageSteps)

        #print("Average steps:", averageSteps)
        #print("Tested rate:", test)

        test /= numberToDivideWith

    # Select the learning rate with the fewest average steps
    best = testRateSteps.index(min(testRateSteps))
    bestRate = testRateInputs[best]

    print("\nBest learning rate:", bestRate)
    print("Device:", device)
    print(
        "Time to find best learning rate:",
        time.perf_counter() - findTimer
    )

    return bestRate
'''
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
            #if s == maxTrainingSteps:
                #print("Failed to converge at learning rate:", test)
            #print("Same rate test ", j)
        #print("Average steps: ",sum / trials)
        testRateSteps.append(sum / trials)
        test /= numberToDivideWith
        #print("Tested rate: ", test)

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

if load:
    weights = loadWeights()


#Initial Guess
print("Max training steps: ",maxTrainingSteps)
print("Learning Rate: ",learningRate)
print("Weights: ", weights)

if showInitialGuesses:
    for i in range(len(inputs)):
        print("Initial Guesses",inputs[i],": ",model(inputs[i]))


trainedMeta = train()
trainingSteps = trainedMeta[0]
trainingTime = trainedMeta[1]


saveWeights(weights)


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
    totalError += abs(a)

print("\nError: ", round(error/totalError * 100,2),"%")
print("Finished training in",trainingTime,"s in ",trainingSteps," steps, meeting the threshold of ",finishThreshold,".")


print("\n--------------Weights & Biases----------------")
print("*First value is bias")
print(weights)

output = []
for i in range(len(inputs)):
    output.append(model(inputs[i]))

if showGraph:
    plt.scatter(inputs,output)
    plt.scatter(inputs,answers)
    plt.show()

print("\n--------------Predict new value----------------")
'''
while True:
    v = float(input("Enter a value to predict output: "))
    normalized = -4 + (v - minInput) * 8 / (maxInput - minInput)
    print("Model output: ",model(normalized))
'''




# Export trained results for ui.py
import json

prediction_rows = []
error = 0
total_error = 0

for i in range(len(inputs)):
    predicted = model(inputs[i])
    actual = answers[i]
    absolute_error = abs(actual - predicted)

    prediction_rows.append({
        "input": float(notNormalizedInputs[i]),
        "actual": float(actual),
        "predicted": float(predicted)
    })

    error += absolute_error
    total_error += abs(actual)

average_error = error / total_error * 100 if total_error else 0

ui_results = {
    "steps": trainingSteps,
    "time": trainingTime,
    "average_error": average_error,
    "threshold": finishThreshold,
    "min_input": float(minInput),
    "max_input": float(maxInput),
    "term_divisor": termDivisor,
    "weights": [float(w) for w in weights],
    "rows": prediction_rows
}

print("UI_RESULTS:", json.dumps(ui_results), flush=True)
print("UI_READY", flush=True)
