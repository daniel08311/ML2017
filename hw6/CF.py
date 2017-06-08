from keras.models import model_from_json
import csv
import numpy as np
import sys

file = sys.argv[1] + "test.csv"
output = sys.argv[2]

def rounds(a):
    if 2.1 > a > 1.9 :
        return 2
    elif 3.1 > a > 2.9:
        return 3
    elif 4.1 > a > 3.9:
        return 4
    elif a < 1.1:
        return 1
    elif a > 4.9:
        return 5
    else:
        return a
        
json_file = open('model_0.84610.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_0.84610.h5")

movies_id_test = []
users_id_test = []
with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        movies_id_test.append(row[2])
        users_id_test.append(row[1])
        
movies_id_test = np.asarray(movies_id_test,dtype=int)
users_id_test = np.asarray(users_id_test,dtype=int)

prediction = loaded_model.predict([movies_id_test, users_id_test])
file = open(output,'w')
file.write("TestDataID,Rating\n")
for i in range(len(prediction)):
    file.write(str(i+1) + "," + str(rounds(prediction[i][0])) + "\n")
file.close()
