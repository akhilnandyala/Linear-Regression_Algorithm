from model import LinearRegression
import numpy as np
import random
import pandas as pd

def randomDimension(features):
    model = LinearRegression()

    # X = np.ndarray(shape = (features, 1))
    X = [None] * features

    r = np.ndarray(shape = (features, 1))
    theta = np.ndarray(shape = (features+1, 1))
    #
    # data = np.ndarray(shape = (5000,features))

    for f in range(0, theta.shape[1]):
        theta[f] = random.randrange(-100, 101, 1)
        # print(theta.shape)
        # print(type(theta))
        # print(theta)

    for i in range(0,5000):
        # print(i)
        X = []
        for f in range(0, features):
            # print(f)
            rn = random.randrange(0.0, 1.0)
            # X.insert(f, rn)
            X.append(rn)
            # print(X.shape)
            # print(type(X))
            # print(X)
            r = X[f] * theta[f]
        r = r + theta[-1] + random.randrange(-20, 20, 1)
        # r = random.randrange(-20, 20, 1)
        model.addSample(X,r)



    alpha = 0.000000003
    numOfSteps = 1000
    loop = 100

    for i in range(0, loop):
        theta, cost = model.fit(alpha, numOfSteps)
        print("theta values", theta,"cost", round(cost, 4))
randomDimension(5)