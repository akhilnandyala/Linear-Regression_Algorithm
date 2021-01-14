from model import LinearRegression
import random


def randomPlane():
    model = LinearRegression()

    a = random.randrange(-100, 101, 1)
    b = random.randrange(-100, 101, 1)
    c = random.randrange(-100, 101, 1)

    print('Target: c =', c, 'a = ', a, 'b = ', b)

    for i in range(0,5000):
        x1 = random.randrange(0, 2, 1)
        x2 = random.randrange(0, 2, 1)
        y = (a * x1) + (b * x2) + (c) + random.randrange(-20, 20, 1)
        model.addSample([x1, x2], y)


    alpha = 0.0001
    numOfSteps = 1000
    loop = 100

    for i in range(0, loop):
        theta, cost = model.fit(alpha, numOfSteps)
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))


randomPlane()