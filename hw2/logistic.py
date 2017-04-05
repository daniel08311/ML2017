# -*- coding: utf-8 -*-
import csv
import numpy
import math
import sys

def sigmoid(X,theta):
	z = numpy.dot(X ,theta)
	return 1/(1+numpy.exp(-z))

def predict(output):
	results = []
	for i in range(len(output)):
		if output[i] > 0.5 :
			results.append(1)
		else:
			results.append(0)
	return results

def accuraccy(Y,y):
	return float(numpy.sum(numpy.logical_not(numpy.logical_xor(Y,y)))) / len(Y)

def standardize(input_arr):
	mean = numpy.mean(input_arr,axis = 0)
	standard = numpy.std(input_arr,axis = 0)
	for i in range(len(standard)):
		if standard[i] == 0 :
			standard[i] = 1 
	return ( input_arr - mean ) / standard

numpy.set_printoptions(precision = 10,suppress=True)

x_train_data = sys.argv[3]
y_train_data = sys.argv[4]
x_test_data = sys.argv[5]
output_dir = sys.argv[6]



X = numpy.loadtxt(open(x_train_data, "rb"), delimiter=",", skiprows=1)
X = standardize(X)
X = numpy.column_stack([X,numpy.ones(X[:,0].shape)])
Y = numpy.loadtxt(open(y_train_data, "rb"), delimiter=",")

X_test = numpy.loadtxt(open(x_test_data, "rb"), delimiter=",", skiprows=1)
X_test = standardize(X_test)
X_test = numpy.column_stack([X_test,numpy.ones(X_test[:,0].shape)])

best_param = numpy.loadtxt(open("best.txt", "rb"))
best_param = best_param.astype(int)

x_train = X[best_param]
y_train = Y[best_param]

iters = 75
gradient_total = numpy.ones(len(X[0,:]))
theta = numpy.zeros(len(X[0,:]))

m = numpy.zeros(len(theta))
v = numpy.zeros(len(theta))
alpha = 0.0003
beta_1 = 0.9
beta_2 = 0.999
epsilon = 0.00000001

for i in range(iters):
	for sample_index in range(len(x_train)):
		fx = sigmoid(x_train[sample_index],theta)
		gradient = (-1.0) * numpy.dot( (y_train[sample_index] - fx) , x_train[sample_index] )  #+ 2*lambdas*theta
		m = beta_1*m + (1-beta_1)*gradient
		v = beta_2*v + (1-beta_2)*(gradient**2)
		m_ = m / (1-beta_1**(i+1))
		v_ = v / (1-beta_2**(i+1))
		theta -= (alpha * m_) / (numpy.sqrt(v_) + epsilon) #(Update parameters)

output= sigmoid(X_test,theta)
prediction = predict(output)

result = open(output_dir,"w")
result.write("id,label\n")
for i in range(len(prediction)):
	result.write(str(i + 1) + "," + str(prediction[i]) + "\n")