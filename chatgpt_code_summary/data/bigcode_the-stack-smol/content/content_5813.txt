import math
import collections
import numpy as np

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def __CheckEvaluationCount(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def CountCorrect(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)

def PredictionDiff(xTestRaw, y, yPredicted):
    
    __CheckEvaluationCount(y, yPredicted)
    __CheckEvaluationCount(xTestRaw, y)

    predictionRange = {}
    for i in range(len(y)):
        predictionRange[xTestRaw[i]] = y[i] - yPredicted[i]

    return predictionRange


def Precision(y, yPredicted):
    numerator = TPCount(y, yPredicted)
    denominator = (numerator + FPCount(y, yPredicted))
    return 0.0 if denominator == 0 else numerator / denominator

def Recall(y, yPredicted):
    numerator = TPCount(y, yPredicted)
    denominator = (numerator + FNCount(y, yPredicted))
    return 0.0 if denominator == 0 else numerator / denominator

def FalseNegativeRate(y, yPredicted):
    numerator = FNCount(y, yPredicted)
    denominator = numerator + TPCount(y, yPredicted)
    return 0.0 if denominator == 0 else numerator / denominator

def FalsePositiveRate(y, yPredicted):
    numerator = FPCount(y, yPredicted)
    denominator = numerator + TNCount(y, yPredicted)
    return 0.0 if denominator == 0 else numerator / denominator

def FNCount(y, yPredicted):
    counter = 0
    for i in range(len(y)):
        if(y[i] == 1 and yPredicted[i] == 0):
            counter += 1

    return counter

def FPCount(y, yPredicted):
    counter = 0
    for i in range(len(y)):
        if(y[i] == 0 and yPredicted[i] == 1):
            counter += 1

    return counter

def TNCount(y, yPredicted):
    counter = 0
    for i in range(len(y)):
        if(y[i] == 0 and yPredicted[i] == 0):
            counter += 1

    return counter

def TPCount(y, yPredicted):
    counter = 0
    for i in range(len(y)):
        if(y[i] == 1 and yPredicted[i] == 1):
            counter += 1

    return counter

def UpperAccRange(Accuracy, n):
    return Accuracy + 1.96 * math.sqrt((Accuracy * (1 - Accuracy) / n))

def LowerAccRange(Accuracy, n):
    return Accuracy - 1.96 * math.sqrt((Accuracy * (1 - Accuracy) / n))

def ConfusionMatrix(y, yPredicted):
    print("                 Predicted Negative | Predicted Positive")
    print("Actual Negative | TN: " + str(TNCount(y, yPredicted)) + "          | FP: " + str(FPCount(y, yPredicted)))
    print("Actual Positive | FN: " + str(FNCount(y, yPredicted)) + "          | TP: " + str(TPCount(y, yPredicted)))


def ExecuteAll(y, yPredicted):
    accuracyVal = Accuracy(y, yPredicted)
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", accuracyVal)
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    print("95% confidence range:", LowerAccRange(accuracyVal, len(y)), "to", UpperAccRange(accuracyVal, len(y)) )
    
