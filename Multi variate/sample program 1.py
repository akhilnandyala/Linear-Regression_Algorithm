from model import LinearRegression

def setPlane():
    model = LinearRegression()

    for i in range(0,1000):

        model.addSample([i, 2 * i], 5 * i)
        model.addSample([2 * i, i], 4 * i)


    alpha = 0.000000006
    numOfSteps = 1000
    loop = 10

    for i in range(0, loop):
        theta, cost = model.fit(alpha, numOfSteps)
        # print(round(theta[0][0],2),round(theta[1][0],2),round(theta[2][0],2), round(cost, 4))
        # print(model.getTheta0())
        # print(model.getTheta1())
        # print(model.getTheta2())
        # print(model.getIteration())
        # print(model.getHypothesis())
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))

setPlane()