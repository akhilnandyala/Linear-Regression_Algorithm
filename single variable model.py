import numpy as np
import pandas as pd

# data = np.array([])
data = []

class LinearRegression :
    C = []
    D = []

    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.theta = np.array([[self.theta0], [self.theta1]])
        self.itercount = 0

    def addSample(self, x, y):
        global C
        global D
        self.C.append(x)
        self.D.append(y)
        self.C1 = np.array(self.C)
        self.D1 = np.array(self.D)
        self.C2 = np.c_[np.ones((len(self.C1), 1)), self.C1]
        self.D2 = np.c_[np.ones((len(self.D1), 0)), self.D1]
        return

    def getSamples(self):
        return self.C2, self.D2


    def getLoss(self, theta, x, y):
        m = len(y)
        predictions = np.dot(x, theta)
        self.cost = np.sum((predictions - y) ** 2) / (m)
        return self.cost

    def fit(self, alpha, numOfSteps):
        self.numOfSteps = numOfSteps
        m = len(self.D2)

        self.itercount = self.itercount+1

        for it in range(numOfSteps):
            self.y_pred = np.dot(self.C2, self.theta)
            self.residuals = self.y_pred - self.D2
            self.gradient_vector = np.dot(self.C2.T, self.residuals)
            self.theta = self.theta - ((alpha / m) * self.gradient_vector)
            self.cost = self.getLoss(self.theta, self.C2, self.D2)
        return self.theta, self.cost


    def getTheta0(self):
        return self.theta[0][0]

    def getTheta1(self):
        return self.theta[1][0]

    def getHypothesis(self):
        return "%s" % round(self.y_pred[1][0])

    def __str__(self):
        return "%s + %s x" % (round(self.theta[0][0], 2), round(self.theta[1][0], 2))

    def getIteration(self):
        return self.numOfSteps*self.itercount

    def getValues(self):
        return self.residuals



