from model import LinearRegression
import random
import matplotlib.pyplot as plt
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

#for diagrams
    X, y , z = model.getSamples()
    temp = z
####

    alpha = 0.001
    numOfSteps = 100
    loop = 50

    for i in range(0,loop):
        theta, cost= model.fit(alpha, numOfSteps)
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.scatter(temp, y)
        plt.plot(temp, round(model.theta[0][0], 2) + round(model.theta[1][0], 2) * temp, '-.b', label=model.__str__())
        plt.legend(loc='upper left')
        plt.show()

randomLine()

