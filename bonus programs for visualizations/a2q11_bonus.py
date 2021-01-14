from model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def setLine():
    model = LinearRegression()

    for i in range(0, 1000):
        model.addSample(i, i)

    alpha = 0.000000003
    numOfSteps = 100
    loop = 50

    for i in range(0,loop):
        theta, cost= model.fit(alpha, numOfSteps)
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.linspace(0, 1000)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.plot(x, x, '-b', label='y=x')
        plt.plot(x, round(theta[0][0], 2) + round(theta[1][0], 2) * x, '-.g', label= model.__str__())
        plt.legend(loc='upper left')
        plt.show()


setLine()