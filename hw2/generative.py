# -*- coding: utf-8 -*-
import csv
import numpy
import math
import sys

def predict(X_test):
	predict = []
	count = 0
	for items in X_test :
		predict_c0 = P_C0 * fx(items,MEAN_C0,SIGMA) / ( (P_C0 * fx(items,MEAN_C0,SIGMA)) + (P_C1 * fx(items,MEAN_C1,SIGMA)) )
		if predict_c0[0][0] > 0.5 :
			predict.append(0)
		else:
			predict.append(1)
		count += 1
	return predict

def fx(x,mean,SIGMA) :
	x = numpy.reshape( x,(len(x),1) ) 
	return ( 1/(2*math.pi)**(53) ) * ( 10.040529337/((-1*numpy.linalg.det(SIGMA))**(0.5)) ) * math.e ** ( -(0.5) * numpy.dot ( numpy.dot ( numpy.transpose( (x-mean) ) ,  numpy.linalg.pinv(SIGMA) )  , (x-mean) ) )

def init_class(class_numb,X,Y,check):
	class_ = numpy.zeros((int(class_numb),len(X[0,:])))
	count = 0 
	for i in range( len (X) ):
		if Y[i] == check :
			class_[count] = X[i]
			count += 1
	return class_

x_train_data = sys.argv[3]
y_train_data = sys.argv[4]
x_test_data = sys.argv[5]
output_dir = sys.argv[6]

X = numpy.loadtxt(open(x_train_data, "rb"), delimiter=",", skiprows=1)
Y = numpy.loadtxt(open(y_train_data, "rb"), delimiter=",")

CLASS0_NUMB = float( len(Y)-numpy.sum(Y) )
CLASS1_NUMB = float( numpy.sum(Y) )

class0 = init_class(CLASS0_NUMB,X,Y,0)
class1 = init_class(CLASS1_NUMB,X,Y,1)

class0 = numpy.transpose(class0)
class1 = numpy.transpose(class1)

P_C0 =  CLASS0_NUMB / (CLASS0_NUMB + CLASS1_NUMB)
P_C1 = 	CLASS1_NUMB / (CLASS0_NUMB + CLASS1_NUMB)

MEAN_C0 = numpy.reshape( numpy.mean(class0,axis = 1),(len(X[0,:]),1) )
MEAN_C1 = numpy.reshape( numpy.mean(class1,axis = 1),(len(X[0,:]),1) )

SIGMA_C0 = numpy.dot ( class0 - MEAN_C0  , numpy.transpose ( class0 - MEAN_C0 ) ) / CLASS0_NUMB
SIGMA_C1 = numpy.dot ( class1 - MEAN_C1  , numpy.transpose ( class1 - MEAN_C1 ) ) / CLASS1_NUMB

SIGMA = SIGMA_C0 * P_C0 + SIGMA_C1*P_C1

X_test = numpy.loadtxt(open(x_test_data, "rb"), delimiter=",", skiprows=1)
prediction = predict(X_test)

result = open(output_dir,"w")
result.write("id,label\n")
for i in range(len(prediction)):
	result.write(str(i + 1) + "," + str(prediction[i]) + "\n")
