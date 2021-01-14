from model import LinearRegression
import random
import numpy as np

def randomLine():
    model = LinearRegression()

    a = random.randrange(-5, 11, 1)
    b = random.randrange(-5, 6, 1)

    print('Target: b =', b, 'a = ', a)

    for i in range(0,500):
        X = random.randrange(-4, 7, 1)
        y = b + a * X + np.random.randn()
        model.addSample(X, y)

    alpha = 0.001
    numOfSteps = 100
    loop = 50

    for i in range(0,loop):
      # for printing the values of theta and the cost, during the 0th iteration.
        if i == 0:
            X, y = model.getSamples()
            initial_cost = model.getLoss(model.theta,X,y)
            print("Current Hypothesis:","%s + %s x ," % (round(model.theta[0][0], 2), round(model.theta[1][0], 2)),"cost =", round(initial_cost, 4))
     ################################################################################

        theta, cost= model.fit(alpha, numOfSteps)
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))

randomLine()

