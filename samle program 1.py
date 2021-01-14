from model import LinearRegression

def setLine():
    model = LinearRegression()

    for i in range(0, 1000):
        model.addSample(i, i)

    alpha = 0.000000003
    numOfSteps = 100
    loop = 50

    for i in range(0,loop):
        if i == 0:
            # for printing the values of theta and the cost, during the 0th iteration.
            X, y = model.getSamples()
            initial_cost = model.getLoss(model.theta,X,y)
            print("Current Hypothesis:","%s + %s x ," % (round(model.theta[0][0], 2), round(model.theta[1][0], 2)),"cost =", round(initial_cost, 4))
            #########################################################################

        theta, cost= model.fit(alpha, numOfSteps)
        # print(theta,round(cost,2))
        # print(model.getTheta0())
        # print(model.getTheta1())
        # print(model.getIteration())
        # print(model.getHypothesis())
        print("Current Hypothesis:", model.__str__(), ",cost =", round(cost, 4))


setLine()