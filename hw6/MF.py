import pickle
import numpy as np
import csv
import sys


file = sys.argv[1] + "test.csv"
output = sys.argv[2]

with open('final_nP.p', 'rb') as handle:
    P = pickle.load(handle)

with open('final_nQ.p', 'rb') as handle:
    Q = pickle.load(handle)

movies_id_test = []
users_id_test = []
with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        movies_id_test.append(row[2])
        users_id_test.append(row[1])

predict = []
for i in zip(users_id_test,movies_id_test):
    predict.append( np.dot(P[int(i[0])-1],Q[int(i[1])-1].T) )#+ bias_P[int(i[0])-1] + bias_Q[int(i[1])-1])

file = open(output,'w')
file.write("TestDataID,Rating\n")
for i in range(len(predict)):
    file.write(str(i+1) + "," + str(predict[i]) + "\n")
file.close()


