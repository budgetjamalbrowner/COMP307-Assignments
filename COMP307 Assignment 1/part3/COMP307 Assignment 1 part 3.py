#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import numpy as np
import random as random
import sys


# %%

file = sys.argv[1]
file = pd.read_csv(file)
# file = pd.read_csv('ionosphere.data')
file.to_csv('ionosphere.csv')


# %%


ionosphereDF = pd.read_csv('ionosphere.csv')


# %%


result = pd.DataFrame('f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 class'.split(" "))
for i in range(len(ionosphereDF)):
    data = pd.DataFrame(ionosphereDF['f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 class'][i].split(" "))
    result = pd.concat([result, data], axis=1)


# %%


ionosphereData = result.transpose()
ionosphereData.columns = ionosphereData.iloc[0]
ionosphereData = ionosphereData.iloc[1:]
ionosphereData = ionosphereData.reset_index(drop=True)
ionosphereDataX = ionosphereData.drop('class', axis = 1)
ionosphereDataY = ionosphereData['class']
ionosphereY = ionosphereDataY.replace(list(set(ionosphereDataY))[1], 1)
ionosphereY = ionosphereY.replace(list(set(ionosphereDataY))[0], 0)


# %%


ionosphereDataX = ionosphereDataX.astype(float)


# %%


class Perceptron: 
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None 
    def perceptronFunction(self, X, weight, bias):
        z = np.dot(X, weight) + bias
        output = 1.0 if z > 0 else 0
        return output
    def fit(self, X, y):
        self.weights = np.array([random.random() for i in range(X.shape[1])])
        self.bias = random.random()
        self.y = y
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                preds = self.perceptronFunction(X[i], self.weights, self.bias)
                # implement lecture algorithm
                if y[i] == preds:
                    pass
                elif y[i] < preds: 
                    self.weights += self.learning_rate*(y[i]-preds) * X[i]
                elif y[i] > preds:
                    self.weights += self.learning_rate*(preds+y[i]) * X[i]
                self.bias += self.learning_rate * (y[i] - preds)
        print("Learned weights: ", self.weights)
        print("Learned bias: ", self.bias) 
    def predict(self, x):
        predictions = []
        for i in range(x.shape[0]):
            y = self.perceptronFunction(x[i], self.weights, self.bias)
            predictions.append(y)
        return predictions


# %%


model = Perceptron()


# %%


model.fit(ionosphereDataX.values, ionosphereY)


# %%


predictions = model.predict(ionosphereDataX.values)


# %%


def accuracyScore(predictions, yTest):
    print("Accuracy for the model is:" + str(np.sum(yTest == predictions)/len(yTest)))
    return np.sum(yTest == predictions)/len(yTest)


# %%


accuracyScore(predictions, ionosphereY)


# %%


import sklearn
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(ionosphereDataX, ionosphereY, test_size=0.2, random_state = 307)


# %%


yTrain = yTrain.reset_index(drop=True)
yTest = yTest.reset_index(drop=True)


# %%


ytrain = yTrain.replace(list(set(yTrain))[1], 1)
ytrain = ytrain.replace(list(set(yTrain))[0], 0)
ytest= yTest.replace(list(set(yTest))[1], 1)
ytest= ytest.replace(list(set(yTest))[0], 0)


# %%


model2 = Perceptron()
model2.fit(xTrain.values, ytrain)


# %%


preds2 = model2.predict(xTest.values)


# %%


accuracyScore(preds2, ytest)


# %%




