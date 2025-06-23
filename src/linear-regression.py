# Jordan Simula-Dubois

import pandas as pd
import numpy as np
from scipy.io import arff


class LinearRegression:
    def __init__(self, dataFile):
        self.data = arff.loadarff(dataFile)
        self.weights = np.random.uniform(0,2,self.data.shape[1])
        self.predictions = np.empty(self.data.shape[1], dtype=int)
        for instance in range(0, self.data.shape[0]):
            self.predictions[instance] = self.makePrediction(self.data[instance])    
        self.alpha = 2 * pow(10, -8)
        self.train()

    def makePrediction(instance):
        pass

    def sumOfSquaredError():
        pass

    def train():
        pass



if __name__ == "__main__":
    pass