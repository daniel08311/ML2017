import numpy as np
import nltk
import pickle
import sys
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import model_from_json

test_file = sys.argv[1]
output = sys.argv[2]

X_test_raw = []
file = open(test_file, 'r', encoding = "utf-8")
for row in file:
    splits = row.split(',',1)
    X_test_raw.append(splits[1])  
file.close()
X_test_raw = X_test_raw[1:]

with open('tokenizer_x.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('tokenizer_y.pickle', 'rb') as handle:
    Y_tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
X_test = tokenizer.texts_to_sequences(X_test_raw)
X_test = pad_sequences(X_test)

json_file = open('0.5205_scithresh0.285.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("0.5205_scithresh0.285.h5")

best_threshold = np.load("threshold.npy")
predict_proba = loaded_model.predict_proba(X_test)
prediction = np.array([[1 if predict_proba[i,j]>=best_threshold[j] else 0 for j in range(predict_proba.shape[1])] for i in range(len(predict_proba))])

dic = Y_tokenizer.word_index
final = []
for i in prediction:
    temp = []
    for k in range(len(i)):
        if i[k] == 1:
            for key, val in dic.items():
                if val == k:
                    temp.append(key)
    final.append(temp)

out = open(output,"w")
out.write("id,tags\n")
for i in range(len(final)):
    string = ""
    for classes in final[i]:
        string += classes + " "
    string = string[:-1]
    string = string.upper()
    out.write(str(i)+","+string+"\n")

