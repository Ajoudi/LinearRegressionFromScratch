# LinearRegressionFromScratch
Developing the linear regression model from scratch using python

Simply put:

Linear regression is the prediction of a dependent variable from an independent variable, or multiple dependent variables (multivariate linear regression). 

The equation of linear regression: 

![first equation](http://latex.codecogs.com/gif.latex?y%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1%20x_1%20&plus;%20%5Cbeta_2%20x_2%20&plus;%20...%20%5Cbeta_n%20x_n)

Where ![variable1](http://latex.codecogs.com/gif.latex?%5Cbeta_0) is the bias, and ![variable2](http://latex.codecogs.com/gif.latex?%5Cbeta_1%20%5Cbeta_2%20.....%20%5Cbeta_n) are the coefficients or the weights, and ![variable4](http://latex.codecogs.com/gif.latex?x1%2Cx2....xn) are the feature variables. What machine learning really is for linear regression is finding the optimimum values for ![variable3](http://latex.codecogs.com/gif.latex?%5Cbeta_0%20....%20%5Cbeta_n) that minimize a corresponding error function. 

For Simple Linear Regression, where we only have ![variable1](http://latex.codecogs.com/gif.latex?%5Cbeta_0), and ![variable1](http://latex.codecogs.com/gif.latex?%5Cbeta_1), The Ordinary Least Sqaures (OLS) optimization is one way of finding the values that minimize the error function. 

# Ordinary Least Squares Method

The error function (the Mean Squared Error):

![MSE](http://latex.codecogs.com/gif.latex?J%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7B1%7D%5E%7Bm%7D%28y-%5Chat%7By%7D%29%5E2)

Deriving the OLS estimator: https://are.berkeley.edu/courses/EEP118/current/derive_ols.pdf

We get two equations:

![beta1prediction](http://latex.codecogs.com/gif.latex?%5Cbeta_1_%7Bpredicted%7D%20%3D%20%5Cfrac%7B%5Csum_%7B1%7D%5E%7BN%7D%28x_i-%5Cbar%7Bx%7D%29%28y_i-%5Cbar%7By%7D%29%7D%7B%5Csum_%7B1%7D%5E%7BN%7D%28x_i-%5Cbar%7Bx%7D%29%5E2%7D)

![beta0prediction](http://latex.codecogs.com/gif.latex?%5Cbeta_0_%7Bpredicted%7D%20%3D%20%5Cbar%7By%7D-%5Cbeta_1_%7Bpredicted%7D%5Cbar%7Bx%7D)
  
  
  
These 2 equations allow us to estimate the bias, and coefficient that would minimize the MSE. 
We simply code these equations in python, and output the results. The code can be found in the repo. 

# Batch Gradient Descent

Another optimization is the Gradient Descent method. Deriving the equation for Gradient descent: https://sebastianraschka.com/faq/docs/linear-gradient-derivative.html

Gradient descent simply traverses through the error function graph iteration by iteration until it reaches a local minimum. 
The error function (also the mean squared error): 

![BGD error](http://latex.codecogs.com/gif.latex?J%28%5Cbeta%29%20%3D%20%5Cfrac%7B1%7D%7B2M%7D%5Csum_%7B1%7D%5E%7BM%7D%28%5Cbeta%5E%7BT%7Dx%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%29%5E2)

A great article that shows the derivation of the Gradient Descent learning equation https://sebastianraschka.com/faq/docs/linear-gradient-derivative.html

The equation that we have to find the optimum coefficients:

![BGD learning](http://latex.codecogs.com/gif.latex?%5Cbeta_j%20%3D%20%5Cbeta_j%20-%20%5Calpha%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7B1%7D%5E%7Bm%7D%28%5Cbeta%5ETx%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D)

The corresponding code can be found in the repo. 

The following blog post https://mubaris.com/2017/09/28/linear-regression-from-scratch/ was of great help in developing this post. 
For more details check out my blog: https://databata.wordpress.com/2018/07/09/100-day-ml-challenge/
