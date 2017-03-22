# -*- coding: utf-8 -*-
import csv
import numpy
import math
import sys

train = sys.argv[1]
test = sys.argv[2]
res = sys.argv[3]
print train,test,res

numpy.set_printoptions(precision = 3,suppress=True)

PM_25 = numpy.zeros((240,24))
PM_10 = numpy.zeros((240,24))
RH = numpy.zeros((240,24))
O3 = numpy.zeros((240,24))
TMP = numpy.zeros((240,24))
WIND = numpy.zeros((240,24))
SO2 = numpy.zeros((240,24))
W_D = numpy.zeros((240,24))


count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

def combine(arr):
	ret_arr = numpy.zeros((12,480))
	horizon = numpy.shape((1,480))
	for i in range( len(arr) ):
		if (i+1) % 20 == 0:
			horizon = numpy.concatenate((horizon,arr[i,:] ),0)
			ret_arr[int(i/20)] = horizon[1:]
			horizon = numpy.shape((1,480))
		else:
			horizon = numpy.concatenate((horizon,arr[i,:] ),0)
			#print horizon,i
	return ret_arr

with open(train, 'rb') as f:
    reader = csv.reader(f)
    std = 0
    for row in reader:
        if row[2] == "PM2.5":
        	PM_25[count_1] = row[3:27]
        	count_1 += 1

        elif row[2] == "O3":
			O3[count_2] = row[3:27]
			count_2 += 1

        elif row[2] == "WIND_SPEED":
			WIND[count_3] = row[3:27]
			count_3 += 1

        elif row[2] == "SO2":
			SO2[count_4] = row[3:27]
			count_4 += 1

        elif row[2] == "WIND_DIREC":
			W_D[count_5] = row[3:27]
			count_5 += 1

PM_25_c =  combine(PM_25)
PM_10_c =  combine(PM_10)
RH_c =  combine(RH)
O3_c =  combine(O3)
TMP_c =  combine(TMP)
WIND_c =  combine(WIND)
SO2_c = combine(SO2)
W_D_c = combine(W_D)

PM_25_t = numpy.zeros((240,9))
PM_10_t= numpy.zeros((240,9))
RH_t = numpy.zeros((240,9))
O3_t = numpy.zeros((240,9))
TMP_t = numpy.zeros((240,9))
WIND_t = numpy.zeros((240,9))
SO2_t = numpy.zeros((240,9))
W_D_t = numpy.zeros((240,9))

count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

with open(test, 'rb') as f:
    reader = csv.reader(f)
    std = 0
    for row in reader:

    	if row[1] == "PM2.5":
        	PM_25_t[count_1] = row[2:]
        	count_1 += 1

        elif row[1] == "O3":
			O3_t[count_2] = row[2:]
			count_2 += 1
        
        elif row[1] == "WIND_SPEED":
			WIND_t[count_3] = row[2:]
			count_3 += 1

        elif row[1] == "SO2":
			SO2_t[count_4] = row[2:]
			count_4 += 1

        elif row[1] == "WIND_DIREC":
			W_D_t[count_5] = row[2:]
			count_5 += 1
hour = 9
Iter = 700
Lambda = 2.5
total_error = numpy.zeros(1)

all_theta = numpy.zeros((471,55))
theta = numpy.zeros(55)
gradient_s_total = 0
gradient_v_total = numpy.zeros(54) 
LR = numpy.full(55,0.015)
	
for i in range(Iter):
	error = 0	
	for j in range(471):
		if j % 3 == 0:
			days = 12
			x = numpy.ones(days)
			x = numpy.column_stack([x,PM_25_c[:,j:j+hour]])
			x = numpy.column_stack([x,O3_c[0*days:(1)*days,j:j+hour]])
			x = numpy.column_stack([x,WIND_c[:,j:j+hour]])
			x = numpy.column_stack([x,W_D_c[:,j:j+hour]])
			x = numpy.column_stack([x,SO2_c[0*days:(1)*days,j:j+hour]])
			x = numpy.column_stack([x,numpy.power(PM_25_c[:,j:j+hour],2)])

			Y = PM_25_c[:,j+hour]
			for sample_index in range(len(x)):	
				gradient_s = (-2) * (Y[sample_index] - (numpy.inner(x[sample_index],theta)))  + 2*Lambda*theta[0]
				gradient_s_total +=   (gradient_s)**2
				gradient_v = (-2) * (Y[sample_index] - (numpy.inner(x[sample_index],theta))) * x[sample_index][1:]  + 2*Lambda*theta[1:]
				gradient_v_total +=  (gradient_v)**2
				theta[0] = theta[0] - (LR[0] / math.sqrt(gradient_s_total)) * gradient_s 
				theta[1:] = theta[1:] - (LR[1:] / numpy.sqrt(gradient_v_total)) * gradient_v
			
			error += numpy.sum(numpy.power((Y - numpy.dot(x,theta)),2))
			all_theta[470] = theta

x = numpy.ones(240)
x = numpy.column_stack([x,PM_25_t[:,9-hour:9]])
x = numpy.column_stack([x,O3_t[:,0:9]])
x = numpy.column_stack([x,WIND_t[:,9-hour:9]])
x = numpy.column_stack([x,W_D_t[:,9-hour:9]])
x = numpy.column_stack([x,SO2_t[:,9-hour:9]])
x = numpy.column_stack([x,numpy.power(PM_25_t[:,9-hour:9],2)])
predict = numpy.zeros(240)
predict= numpy.dot(x,all_theta[470])

result = open(res,"w")
result.write("id,value\n")
count = 0 
for prediction in predict:
	result.write("id_"+ str(count)+","+str(prediction)+"\n")
	count += 1
