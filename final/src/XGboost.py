
# coding: utf-8

# In[603]:

# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

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

def predict(test,train,model):
    prediction = []
    arr = list(test[0])
    arr.append(train[len(train)-4,-1])
    arr.append(train[len(train)-3,-1])
    arr.append(train[len(train)-2,-1])
    arr.append(train[len(train)-1,-1])
    arr = np.asarray(arr,dtype=float)
    a0 = model.predict(xgb.DMatrix(arr.reshape((1,len(arr)))))[0]
    prediction.append(a0)

    arr = list(test[1])
    arr.append(train[len(train)-3,-1])
    arr.append(train[len(train)-2,-1])
    arr.append(train[len(train)-1,-1])
    arr.append(a0)
    arr = np.asarray(arr,dtype=float)
    a1 = model.predict(xgb.DMatrix(arr.reshape((1,len(arr)))))[0]
    prediction.append(a1)

    arr = list(test[2])
    arr.append(train[len(train)-2,-1])
    arr.append(train[len(train)-1,-1])
    arr.append(a0)
    arr.append(a1)
    arr = np.asarray(arr,dtype=float)
    a2 = model.predict(xgb.DMatrix(arr.reshape((1,len(arr)))))[0]
    prediction.append(a2)

    arr = list(test[3])
    arr.append(train[len(train)-1,-1])
    arr.append(a0)
    arr.append(a1)
    arr.append(a2)
    arr = np.asarray(arr,dtype=float)
    a3 = model.predict(xgb.DMatrix(arr.reshape((1,len(arr)))))[0]
    prediction.append(a3)

    last = [a0,a1,a2,a3]

    for i in(range(4,len(test))):
        arr = list(test[i])
        arr.append(last[0])
        arr.append(last[1])
        arr.append(last[2])
        arr.append(last[3])
        arr = np.asarray(arr,dtype=float)
        p = model.predict(xgb.DMatrix(arr.reshape((1,len(arr)))))[0]
        prediction.append(p)
        last = last[1:]
        last.append(p)
    return prediction

THRESHOLD = [0.05]
TRAIN_PATH = sys.argv[1]
TEST_PATH = sys.argv[2]
LABEL_PATH = sys.argv[3]
OUTPUT_PATH = sys.argv[4]

total = np.zeros((416,1))
for thresh in THRESHOLD:
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.drop('week_start_date',1)
    train_df = train_df.replace("sj",0)
    train_df = train_df.replace("iq",1)
    train_df = train_df.fillna(train_df.mean())
    week = np.array(train_df['weekofyear'],dtype=float)
    train_df['week_sin'] = np.sin(week / 52) * np.pi
    train_df = train_df.drop('weekofyear',1)
    X_train = train_df.values.astype(float)

    test_df = pd.read_csv(TEST_PATH)
    test_df = test_df.drop('week_start_date',1)
    test_df = test_df.replace("sj",0)
    test_df = test_df.replace("iq",1)
    test_df = test_df.fillna(test_df.mean())
    week = np.array(test_df['weekofyear'],dtype=float)
    test_df['week_sin'] = np.sin(week / 52) * np.pi
    test_df = test_df.drop('weekofyear',1)
    test = test_df.values.astype(float)

    All = np.append(X_train,test,axis=0)

    labels_df = pd.read_csv(LABEL_PATH)
    labels = labels_df.values
    Y_train = labels[:,-1].astype(float)

    X_train = All[:1456]
    new = []
    feats = []
    for i in range(1,len(X_train[0])):
        if abs(np.corrcoef(Y_train, X_train[:,i]))[0][1] > thresh:
            new.append(All[:,i])
            if abs(np.corrcoef(Y_train, X_train[:,i]))[0][1] > thresh:
                feats.append(len(new)-1)
    feats = feats[1:]

    new = np.asarray(new,dtype=float).T
    all_sj = np.append(new[:936],new[1456:1716],axis=0)
    all_iq = np.append(new[936:1456],new[1716:],axis=0)

    scaler = StandardScaler()    
    all_sj = scaler.fit_transform(all_sj)
    all_iq = scaler.fit_transform(all_iq)

    X_train_sj = all_sj[:936]
    X_train_iq = all_iq[:520]

    test_sj = all_sj[936:]
    test_iq = all_iq[520:]

    Y_train_sj = Y_train[:936]
    Y_train_iq = Y_train[936:]

    test_sj = combine_feats_test(test_sj,X_train_sj)
    test_iq = combine_feats_test(test_iq,X_train_iq)
    X_train_sj = combine_feats(X_train_sj,Y_train_sj)
    X_train_iq = combine_feats(X_train_iq,Y_train_iq)

    X_train, X_test, Y_train, Y_test = train_test_split(X_train_sj, Y_train_sj, test_size=0.001, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    params = {"objective": "reg:linear", "booster":"gbtree", 'max_depth':'4', 'eta':'0.02', 'subsample':'0.7', 'eval_metric':'mae'}
    params['nthread'] = 8   
    evallist  = [(dtest,'eval')]
    num_round = 400
    gbm_1 = xgb.train(params, dtrain, num_round)  

    X_train, X_test, Y_train, Y_test = train_test_split(X_train_iq, Y_train_iq, test_size=0.001, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    params = {"objective": "reg:linear", "booster":"gbtree", 'max_depth':'4', 'eta':'0.02', 'subsample':'0.7', 'eval_metric':'mae'}
    params['nthread'] = 8
    evallist  = [(dtest,'eval')]
    num_round = 400
    gbm_2 = xgb.train(params, dtrain, num_round)

    predict_sj = predict(test_sj,X_train_sj,gbm_1)
    predict_iq = predict(test_iq,X_train_iq,gbm_2)
    predict_all = predict_sj+predict_iq
    predict_all = np.asarray(predict_all,dtype=float)
    total += predict_all.reshape((len(predict_all),1))

total /= len(THRESHOLD)
output = open(OUTPUT_PATH,'w')
output.write("city,year,weekofyear,total_cases\n")
for i in range(len(test)):
    if test[i][0] == 0:
        output.write("sj," + str(test[i][1]) + "," + str(week[i]) + "," + str(int(round(total[i][0]))) + "\n")
    else:
        output.write("iq," + str(test[i][1]) + "," + str(week[i]) + "," + str(int(round(total[i][0]))) + "\n")




