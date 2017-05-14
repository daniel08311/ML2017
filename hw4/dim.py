import numpy as np
from keras.constraints import maxnorm
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold
from keras.layers import Dropout
import csv
import keras,sys
from keras import backend as K
from sklearn import preprocessing
K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data

def run_keras(X_train,Y_train):
    model = Sequential()
    model.add(Dense(60, activation='relu',input_dim=200))
    model.add(Dropout(0.15))
    model.add(Dense(30, activation='relu',kernel_constraint=maxnorm(3.)))
    model.add(Dropout(0.15))
    model.add(Dense(10, activation='relu',kernel_constraint=maxnorm(3.)))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation='relu'))
    print (model.summary())
    opt = keras.optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_logarithmic_error',optimizer=opt)#,metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=30000,epochs=3200)
    return model


npz = sys.argv[1]
output = sys.argv[2]

SET = 2000
# if we want to generate data with intrinsic dimension of 10
means = np.zeros((60,SET,200))
for dim in range(60):
    for i in range(SET):
        N = 2500
        # the hidden dimension is randomly chosen from [60, 79] uniformly
        layer_dims = [np.random.randint(60, 80), 100]
        datas = gen_data(dim+1, layer_dims, N)
        var = np.var(datas,axis = 0).reshape((1,100))
        mean = np.mean(datas,axis = 0).reshape((1,100))
        col = np.concatenate((var,mean),axis = 1)
        means[dim][i] = col[0]

X_train = means[0]
for i in means[1:]:
    X_train = np.vstack((X_train,i))
min_max_scaler = preprocessing.MinMaxScaler() 
X_train = preprocessing.scale(X_train)

Y_train = np.zeros(SET*60)
for i in range(60):
    Y_train[SET*i:SET*(i+1)] = i+1


result = run_keras(X_train,Y_train)

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



