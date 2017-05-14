import numpy as np
from sklearn import preprocessing
from keras.models import model_from_json, load_model
import csv
import keras,sys
from keras import backend as K
K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")

npz = sys.argv[1]
output = sys.argv[2]

result = load_model("model.h5")
data = np.load(npz)
X_test = np.zeros((200,200))
for i in range(200):
    x = data[str(i)]
    mean = np.mean(x , axis=0).reshape((1,100))
    var = np.var(x , axis=0).reshape((1,100))
    col = np.concatenate((var,mean),axis = 1)
    X_test[i] = col
X_test = preprocessing.scale(X_test)

predict = result.predict(X_test)
predict = np.log(predict)
csv = open(output,"w")
csv.write("SetId,LogDim\n")
for i in range(len(predict)):
    csv.write(str(i) + "," + str(predict[i][0]) + "\n")
csv.close()