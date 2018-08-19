import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


def SimpleLinearRegressionOLS(X_train, Y_train):
	X_mean = np.mean(X_train)
	Y_mean = np.mean(Y_train)
	sum_num = 0
	sum_bot = 0
	predict_slope = 0
	predict_bias = 0 

	for i in range(0,len(X_train)):
		top_Result = (X_train[i] - X_mean)*(Y_train[i]-Y_mean)
		bot_Result = (X_train[i] - X_mean)**2
		sum_num = top_Result + sum_num
		sum_bot = bot_Result + sum_bot 

	predict_slope = sum_num/sum_bot
	predict_bias = Y_mean - (predict_slope*X_mean)
	
	return predict_slope,predict_bias

def SimpleLinearRegressionOLSPredict(X_test,predict_slope, predict_bias):
	y_Predict = 0
	y_PredictedList = []
	for i in range(0,len(X_test)):
		y_Predict = predict_slope*X_test[i] + predict_bias
		y_PredictedList += [y_Predict]
	return y_PredictedList

def SimpleLinearRegressionEvaluateMSE(Y_predict, Y_test):
	error_Sum = 0
	MSE = 0
	for i in range(0,len(Y_test)):
		temp_sum = 0
		temp_sum = Y_test[i] - Y_predict[i]
		error_Sum = (temp_sum**2) + error_Sum
	MSE = error_Sum/len(Y_test)

	return MSE


if __name__ == '__main__':

	slope = 0
	bias = 0
	y_predicted = []
	MSE = 0

	dataset = pd.read_csv('studentscores.csv')
	X = dataset['Hours'].values
	Y = dataset['Scores'].values

	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)

	slope,bias = SimpleLinearRegressionOLS(X_train,Y_train)
	y_predicted = SimpleLinearRegressionOLSPredict(X_test,slope,bias)

	MSE = SimpleLinearRegressionEvaluateMSE(y_predicted,Y_test)
	print(MSE)