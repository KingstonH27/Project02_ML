#https://chatgpt.com/share/6abea1dc-3638-83e9-b2ca-0f186a86d448
#https://chatgpt.com/share/6ac52c9a-7678-83e9-b8cc-8825801321dd
#https://chatgpt.com/share/6ace8d81-2c30-83e9-be64-d4abaa1b0d79


import pandas as pd
import random
import matplotlib.pyplot as plt
import sympy as sp
import time

RESET = '\033[0m'



DATA = pd.read_csv("housing.csv")
DATA = DATA.sort_values("median_house_value")




def getData(inputColumn,rows):
   return DATA.iloc[:rows,inputColumn].values.tolist()
def getAnswers(inputColumn,rows):
   return DATA.iloc[:rows,inputColumn].values.tolist()



answerCol = 9
inputCol = 4
rowCount = 5

DATAinputs = getData(inputCol,rowCount)
Dataanswers = getAnswers(inputCol,rowCount)

finishThreshold = 0.5
maxTrainingSteps = 100
learningRate = 0.0001

#inputs = DATAinputs
#answers = Dataanswers
#print("hi")
#print(inputs)
inputs = [2,3]
answers = [2,6]
#First input doesn't do anything, its the input compensating for weights list length(due to bias)
#Inputs = [bias,x^1,x^2,x^3]
#inputs = [[0,7,1,1]]
#answers = [28,1.2,3]



weights = [random.randint(-10, 10)/10] + [random.randint(-10, 10)/10 for _ in range(3)]

#Putting learning rate at 0.5 will result in exact answers when only trained on one set of inputs because it sets the bias to the target number


def model(ins):

    y = 0
    for term in range(4):

        y += weights[term] * ins**term
    return y


b,w1,w2,w3= sp.symbols("b,w1,w2,w3")
x = sp.symbols("x")
y = sp.symbols("y")

weightSymbols = [b,w1,w2,w3]


prediction = b + w1 * x + w2* x**2 + w3*x**3
loss = (y - prediction) ** 2

derivatives = []
for i in range(4):
    derivatives.append(sp.diff(loss, weightSymbols[i]))

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


def findBestLearningRate():
    findTimer = time.perf_counter()
    global inputs, answers, weights, learningRate
    startingInputs = inputs.copy()
    startingAnswers = answers.copy()
    startingWeights = weights.copy()
    testRateInputs = []
    testRateSteps = []
    test = 1.0
    for i in range(6):
        testRateInputs.append(test)
        sum = 0
        for j in range(10):
            inputs = startingInputs.copy()
            answers = startingAnswers.copy()
            weights = startingWeights.copy()
            learningRate = test
            s, t = train()
            sum += s
            #print("Same rate test ", j)
        print(sum / 10)
        testRateSteps.append(sum / 10)
        test /= 10
        print("Tested rate ", test)

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
    p = model(inputs[i])
    a = answers[i]
    PREDICTEDCOLOR = '\033[92m' if abs(a-p)<finishThreshold else '\033[0m'
    print(f"{inputs[i]:<12} | {PREDICTEDCOLOR}{p:<15}{RESET}| {a:<15}")

    error += round(abs(a-p),3)
    totalError += a

print("\nError: ", round(round(error,4)/round(totalError,4) * 100,2),"%")
print("Finished training in",trainingTime,"s in ",trainingSteps," steps, meeting the threshold of ",finishThreshold,".")


print("\n--------------Weights & Biases----------------")
print(weights)


#plt.scatter(x,y)

plt.show()