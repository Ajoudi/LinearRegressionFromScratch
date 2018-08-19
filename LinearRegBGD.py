import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

def cost_function(X, Y, B):
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * len(Y))
    return J

def gradientDescent(X,Y,B,alpha,iterations):
	cost_history = [0] * iterations
	for iteration in range(iterations):
		h = X.dot(B)
		loss = h - Y
		gradient = X.T.dot(loss) / len(Y)
		B = B - alpha * gradient
		cost = cost_function(X, Y, B)
		cost_history[iteration] = cost
	return B, cost_history

def LinearRegressionPrediction(B,X):
	Y_predicted = B.dot(X)
	return Y_predicted

def SimpleLinearRegressionEvaluateMSE(Y_predict, Y_test):
	error_Sum = 0
	MSE = 0
	for i in range(0,len(Y_test)):
		temp_sum = 0
		temp_sum = Y_test[i] - Y_predict[i]
		error_Sum = (temp_sum**2) + error_Sum
	MSE = error_Sum/len(Y_test)

	return MSE


if __name__== '__main__':
	
	dataset = pd.read_csv('studentscores.csv')
	X = dataset['Hours'].values
	Y = dataset['Scores'].values

	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)

	X1 = np.array([np.ones(len(X_train)),X_train]).T 
	B = np.zeros(2)
	alpha = 0.0001
	
	newB, cost_history = gradientDescent(X1, Y_train, B, alpha, 10000)

	X2 = np.array([np.ones(len(X_test)),X_test])
	y_predicted = LinearRegressionPrediction(newB,X2)


	MSE = SimpleLinearRegressionEvaluateMSE(y_predicted, Y_test)

	print(MSE)