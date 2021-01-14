import numpy as np
import pandas as pd

class LinearRegression:
    C = []
    D = np.array([])
    E = []
    data = []

    def __init__(self):
        self.itercount = 0

    def addSample(self, x, z):
        global C
        global D
        global data

        self.C.append(x)
        self.D = np.array(self.C)
        self.D = np.c_[np.ones((len(self.D), 1)), self.D]
        self.E.append(z)
        self.E1 = np.array(self.E)
        self.E1 = np.c_[np.ones((len(self.E1), 0)), self.E1]
        self.theta = np.ndarray(shape=(self.D.shape[1], 1))

        return


    def getSamples(self):
        self.D = np.array(self.C)
        self.D = np.c_[np.ones((len(self.D), 1)), self.D]
        self.E1 = np.c_[np.ones((len(self.E1), 0)), self.E1]
        return self.D, self.E1

    def getLoss(self, data, theta):
        m = len(self.E1)
        predictions = np.dot(data, theta)
        self.cost = np.sum((predictions - self.E1) ** 2) / (m)
        return self.cost


    def fit(self, alpha, numOfSteps):
        self.numOfSteps = numOfSteps
        m = len(self.D)

        self.itercount = self.itercount+1

        for it in range(numOfSteps):
            self.y_pred = np.dot(self.D, self.theta)
            self.residuals = self.y_pred - self.E1
            self.gradient_vector = np.dot(self.D.T, self.residuals)
            self.theta = self.theta - ((alpha / m) * self.gradient_vector)
            self.cost = self.getLoss(self.D, self.theta)
        return self.theta, self.cost

    def getTheta0(self):
        return self.theta[0][0]

    def getTheta1(self):
        return self.theta[1][0]

    def getTheta2(self):
        return self.theta[2][0]

    def getHypothesis(self):
        return "%s" % round(self.y_pred[1][0])

    def __str__(self):
        return "%s + %s x1 + %s x2" % (round(self.theta[0][0], 3), round(self.theta[1][0], 3), round(self.theta[2][0], 2))

    def getIteration(self):
        return self.numOfSteps*self.itercount

    def getValues(self):
        return self.residuals

