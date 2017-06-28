
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import keras
import matplotlib.pylab as plt
from keras.models import Sequential
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1,l2

# In[2]:

def combine_feats(arr,Y):
    new_arr = []
    for i in range(len(arr)):
        features = list(arr[i])
        if i == 0 :
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i])
            features.append(Y[i])
            features.append(Y[i])
            features.append(Y[i])

        elif i == 1:
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i-1])
            features.append(Y[i-1])
            features.append(Y[i-1])
            features.append(Y[i-1])

        elif i == 2:
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i-2])
            features.append(Y[i-2])
            features.append(Y[i-2])
            features.append(Y[i-1])

        elif i == 3:
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i-3])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])


        elif i == 4:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])

        elif i == len(arr)-5:
            for feat_idx in feats:
                features.append(arr[i-5,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])

        elif i == len(arr)-4:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])
       
        elif i == len(arr)-3:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])

        elif i == len(arr)-2:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])

        elif i == len(arr)-1:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])

        else:
            for feat_idx in feats:
                features.append(arr[i-5,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
            features.append(Y[i-4])
            features.append(Y[i-3])
            features.append(Y[i-2])
            features.append(Y[i-1])
        new_arr.append(features)
    return np.asarray(new_arr,dtype=float)

def combine_feats_test(arr,train):
    new_arr = []
    for i in range(len(arr)):
        features = list(arr[i])
        if i == 0 :
            for feat_idx in feats:
                features.append(train[len(train)-5,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-4,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-3,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-2,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])

        elif i == 1:
            for feat_idx in feats:
                features.append(train[len(train)-4,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-3,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-2,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])

        elif i == 2:
            for feat_idx in feats:
                features.append(train[len(train)-3,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-2,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])


        elif i == 3:
            for feat_idx in feats:
                features.append(train[len(train)-2,feat_idx])
            for feat_idx in feats:
                features.append(train[len(train)-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])


        elif i == 4:
            for feat_idx in feats:
                features.append(train[len(train)-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])

        elif i == len(arr)-5:
            for feat_idx in feats:
                features.append(arr[i-5,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])

        elif i == len(arr)-4:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
       
        elif i == len(arr)-3:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])

        elif i == len(arr)-2:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])

        elif i == len(arr)-1:
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])
            for feat_idx in feats:
                features.append(arr[i,feat_idx])

        else:
            for feat_idx in feats:
                features.append(arr[i-5,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i-1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+1,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+2,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+3,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+4,feat_idx])
            for feat_idx in feats:
                features.append(arr[i+5,feat_idx])
        new_arr.append(features)
    return np.asarray(new_arr,dtype=float)

np.set_printoptions(precision=5,suppress=True)
np.random.seed(0)
train_df = pd.read_csv("train.csv")
train_df = train_df.replace("sj",0)
train_df = train_df.replace("iq",1)
train_df = train_df.fillna(train_df.mean())
week = np.array(train_df['weekofyear'],dtype=float)
train_df['week_sin'] = np.sin(week / 52) * np.pi
train_df = train_df.drop('weekofyear',1)
X_train = train_df.values.astype(float)

test_df = pd.read_csv("test.csv")
test_df = test_df.replace("sj",0)
test_df = test_df.replace("iq",1)
test_df = test_df.fillna(test_df.mean())
week = np.array(test_df['weekofyear'],dtype=float)
test_df['week_sin'] = np.sin(week / 52) * np.pi
test_df = test_df.drop('weekofyear',1)
test = test_df.values.astype(float)

All = np.append(X_train,test,axis=0)

# scaler = StandardScaler()#feature_range=(0, 1))
# All = scaler.fit_transform(All)

labels_df = pd.read_csv("dengue_labels_train.csv")
labels = labels_df.values
Y_train = labels[:,-1].astype(float)

X_train = All[:1456]
new = []
feats = []
for i in range(1,len(X_train[0])):
    if abs(np.corrcoef(Y_train, X_train[:,i]))[0][1] > 0.0:
        new.append(All[:,i])
        if abs(np.corrcoef(Y_train, X_train[:,i]))[0][1] > 0.0:
            feats.append(len(new)-1)
feats = feats[1:]

new = np.asarray(new,dtype=float).T
all_1 = np.append(new[:936],new[1456:1716],axis=0)
all_2 = np.append(new[936:1456],new[1716:],axis=0)

scaler = StandardScaler()#feature_range=(0, 1))
all_1 = scaler.fit_transform(all_1)
all_2 = scaler.fit_transform(all_2)

X_train_1 = all_1[:936]
X_train_2 = all_2[:520]

test_1 = all_1[936:]
test_2 = all_2[520:]

Y_train_1 = Y_train[:936]
Y_train_2 = Y_train[936:]

test_1 = combine_feats_test(test_1,X_train_1)
test_2 = combine_feats_test(test_2,X_train_2)
X_train_1 = combine_feats(X_train_1,Y_train_1)
X_train_2 = combine_feats(X_train_2,Y_train_2)

# test_1 = (test_1)
# test_2 = (test_2)

total_1 = np.zeros((260,))
total_2 = np.zeros((156,))
for n in range(1):
	print("="*50,n)
	model = Sequential()
	model.add(Dense(1024, input_dim=X_train_1.shape[1], kernel_regularizer=l1(0.01)))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(512))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(256))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(128))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(64))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(1))
	opt = keras.optimizers.adam(lr=0.001)
	model.compile(loss='MAE', optimizer=opt)
	filepath="dnn_temp_1.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True, mode='min')
	earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
	callbacks_list = [earlystop,checkpoint]
	model.fit(X_train_1, Y_train_1, validation_split=0.001, epochs = 1000, batch_size = 5000, verbose=1, callbacks = callbacks_list )


	model = Sequential()
	model.add(Dense(1024, input_dim=X_train_1.shape[1], kernel_regularizer=l1(0.01)))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(512))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(256))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(128))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(64))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(1))
	opt = keras.optimizers.adam(lr=0.001)
	model.compile(loss='MAE', optimizer=opt)
	model.load_weights("dnn_temp_1.hdf5")

	trainPredict_sj_val = np.clip(np.ravel(model.predict(X_train_1)), a_min=0, a_max=1000)
	plt.figure(figsize=(10,5))
	plt.plot(Y_train_1)
	plt.plot(trainPredict_sj_val)
	plt.savefig('sj.png')

	#city-iq train
	#X_train, X_test, Y_train, Y_test = train_test_split(X_train_2, Y_train_2, test_size=0.1, random_state=n)
	model_2 = Sequential()
	model_2.add(Dense(1024, input_dim=X_train_1.shape[1], kernel_regularizer=l1(0.01)))
	model_2.add(PReLU(alpha_initializer='zero',weights=None))
	model_2.add(Dropout(0.1))
	model.add(Dense(512))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(256))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(128))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(64))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model_2.add(Dense(1))
	opt = keras.optimizers.adam(lr=0.001)
	model_2.compile(loss='MAE', optimizer=opt)
	filepath="dnn_temp.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True, mode='min')
	earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
	callbacks_list = [earlystop,checkpoint]
	model_2.fit(X_train_2, Y_train_2, validation_split = 0.001, epochs = 1000, batch_size = 5000, verbose=1, callbacks = callbacks_list )


	model_2 = Sequential()
	model_2.add(Dense(1024, input_dim=X_train_1.shape[1], kernel_regularizer=l1(0.01)))
	model_2.add(PReLU(alpha_initializer='zero',weights=None))
	model_2.add(Dropout(0.1))
	model.add(Dense(512))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(256))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(128))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model.add(Dense(64))
	model.add(PReLU(alpha_initializer='zero',weights=None))
	model.add(Dropout(0.1))
	model_2.add(Dense(1))
	opt = keras.optimizers.adam(lr=0.001)
	model_2.compile(loss='MAE', optimizer=opt)
	model_2.load_weights("dnn_temp.hdf5")

arr = list(test_1[0])
arr.append(X_train_1[len(Y_train_1)-4,-1])
arr.append(X_train_1[len(Y_train_1)-3,-1])
arr.append(X_train_1[len(Y_train_1)-2,-1])
arr.append(X_train_1[len(Y_train_1)-1,-1])
arr = np.asarray(arr,dtype=float)
a0 = model.predict((arr.reshape((1,len(arr)))))[0][0]
print (a0)

arr = list(test_1[1])
arr.append(X_train_1[len(Y_train_1)-3,-1])
arr.append(X_train_1[len(Y_train_1)-2,-1])
arr.append(X_train_1[len(Y_train_1)-1,-1])
arr.append(a0)
arr = np.asarray(arr,dtype=float)
a1 = model.predict((arr.reshape((1,len(arr)))))[0][0]
print (a1)

arr = list(test_1[2])
arr.append(X_train_1[len(Y_train_1)-2,-1])
arr.append(X_train_1[len(Y_train_1)-1,-1])
arr.append(a0)
arr.append(a1)
arr = np.asarray(arr,dtype=float)
a2 = model.predict((arr.reshape((1,len(arr)))))[0][0]
print (a2)

arr = list(test_1[3])
arr.append(X_train_1[len(Y_train_1)-1,-1])
arr.append(a0)
arr.append(a1)
arr.append(a2)
arr = np.asarray(arr,dtype=float)
a3 = model.predict((arr.reshape((1,len(arr)))))[0][0]
print (a3)

last = [a0,a1,a2,a3]

for i in(range(4,len(test_1))):
    arr = list(test_1[i])
    arr.append(last[0])
    arr.append(last[1])
    arr.append(last[2])
    arr.append(last[3])
    arr = np.asarray(arr,dtype=float)
    p = model.predict((arr.reshape((1,len(arr)))))[0][0]
    print(p)
    last = last[1:]
    last.append(p)

arr = list(test_2[0])
arr.append(X_train_2[len(Y_train_2)-4,-1])
arr.append(X_train_2[len(Y_train_2)-3,-1])
arr.append(X_train_2[len(Y_train_2)-2,-1])
arr.append(X_train_2[len(Y_train_2)-1,-1])
arr = np.asarray(arr,dtype=float)
a0 = model_2.predict((arr.reshape((1,len(arr)))))[0][0]
print (a0)

arr = list(test_2[1])
arr.append(X_train_2[len(Y_train_2)-3,-1])
arr.append(X_train_2[len(Y_train_2)-2,-1])
arr.append(X_train_2[len(Y_train_2)-1,-1])
arr.append(a0)
arr = np.asarray(arr,dtype=float)
a1 = model_2.predict((arr.reshape((1,len(arr)))))[0][0]
print (a1)

arr = list(test_2[2])
arr.append(X_train_2[len(Y_train_2)-2,-1])
arr.append(X_train_2[len(Y_train_2)-1,-1])
arr.append(a0)
arr.append(a1)
arr = np.asarray(arr,dtype=float)
a2 = model_2.predict((arr.reshape((1,len(arr)))))[0][0]
print (a2)

arr = list(test_2[3])
arr.append(X_train_2[len(Y_train_2)-1,-1])
arr.append(a0)
arr.append(a1)
arr.append(a2)
arr = np.asarray(arr,dtype=float)
a3 = model_2.predict((arr.reshape((1,len(arr)))))[0][0]
print (a3)

last = [a0,a1,a2,a3]

for i in(range(4,len(test_2))):
    arr = list(test_2[i])
    arr.append(last[0])
    arr.append(last[1])
    arr.append(last[2])
    arr.append(last[3])
    arr = np.asarray(arr,dtype=float)
    p = model_2.predict((arr.reshape((1,len(arr)))))[0][0]
    print(p)
    last = last[1:]
    last.append(p)