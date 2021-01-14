# Linear-Regression_Algorithm
Linear Regression algorithm development for single variable and multi variate problems

Methods of the class LinearRegression:
 The constructor has no other parameters, other than self.
 The class has a method addSample, which is used to an example, x, and its value, y. In order to add N examples to a
LinearRegression, there will be N calls to the method addSample.
 Once all the examples have been provided, the method fit can be called. That method has two parameters: the 'alpha' to use,
and the number of iterations to be performed during that call of the method.
 getHypothesis is an instance method receiving an example as input and returning the (value) value of the hypothesis
evaluated on the given example, in other workds, h-theta(x).
 str(self) returns a string representation of the hypothesis function represented by the LinearRegression object. E.g.:
4.03 + 4.09 x
 getLoss returns the mean squared error for the current set of examples and the current values of theta0 and theta1.
 getTheta0 returns the value of theta0.
 getTheta1 returns the value of theta1.
 getIteration returns the current number of iterations. This value is incremented by 1 for each step performed by the
method fit. Specifically, after a call model.fit(0.003, 100), model.getIteration() should return 100, after a secon call,
model.fit(0.003, 100), model.getIteration() should return 200, etc.
 getSamples returns the list of all the examples.
 getValues returns the list of all the labels. The order of the elements is the same as for getSamples
