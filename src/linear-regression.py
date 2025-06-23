# Jordan Simula-Dubois

import pandas as pd
import numpy as np
from scipy.io import arff
import math

ALPHA = 2e-8


class LinearRegression:
    def __init__(self, dataFile):
        data = arff.loadarff(dataFile)
        self.predictors = data.iloc[:, :-1]
        self.labels = data.iloc[:, -1]
        self.weights = np.random.uniform(0,2,self.predictors.shape[1] + 1)
        self.predictions = np.empty(self.predictors.shape[0], dtype=int)
        for instance in range(0, self.predictors.shape[0]):
            self.predictions[instance] = self.makePrediction(self.predictors.iloc[instance])    
        self.alpha = ALPHA
        self.train()


    def makePrediction(instance):
        sum = self.weights[0]
        for feature in range(0, instance.shape[1]):
           sum += instance[feature] * self.weights[feature + 1]
        return sum


    def sumOfSquaredError():
        sum = 0.0
        for instance in range(0, self.predictors.shape[0]):
            sum += pow((self.labels[instance][0] - self.predictions[instance]), 2)
        return sum/2


    def errorDelta(weightId):
        sum = 0.0
        for instance in range(0,self.predictor.shape[0]):
            if weightId == 0:
                sum += self.label[instance][0] - self.prediction
            else:
                sum += (self.labels[instance][0] - self.predictions[instance]) * self.predictors[instance][weightId-1]
        return sum


    def train():
        sse = self.sumOfSquaredError()
        iteration = 0
        minimalProgress = 0

        while(noProgressCounter < 100):

            for weight in range(0, self.weights.size):
                self.weights[weight] = self.weights[weight] + self.alpha * self.errorDelta(weight)
            for instance in range(self.predictors.shape[0]):
                self.predictions[instance] = self.makePrediction(self.predictors.iloc[instance])

            newSSE = self.sumOfSquaredError()

            if(math.isclose(sse, newSSE, 1e-9)):
                minimalProgress += 1
            else:
                minimalProgress = 0
            
            sse = newSSE

            if(iteration % 1000000) == 0:
                print("iteration " + str(i) + ":")
                print("weights: " + np.array_str(self.weights))
                print("Sum of Squared Error: " + str(sse))
                print()
            
            iteration += 1

        print("Training has come to a stop at iteration " + str(iteration))
        print("Final weights: " + np.array_str(self.weights))
        print("Final Sum of Squared Error: " + str(sse))

        return



if __name__ == "__main__":
    pass