import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Nadam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

np.random.seed(7)

data = pd.read_csv('train.csv', index_col=[0, 1, 2])
label = pd.read_csv('label.csv', index_col=[0, 1, 2])
test_data = pd.read_csv('test.csv', index_col=[0, 1, 2])

features = ['reanalysis_dew_point_temp_k',
            'reanalysis_min_air_temp_k',
            'reanalysis_air_temp_k',
            'reanalysis_avg_temp_k',
            'week_sine',
            'reanalysis_relative_humidity_percent',
            'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_tdtr_k',
            ]

def processing(df):
    df = df.drop('week_start_date',1)
    df.interpolate()
    df = df.fillna(method='ffill')
    
    week = df.index.get_level_values('weekofyear')
    df['week_sine'] = np.sin(week / 52) * np.pi
    
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df[features].values

def batch_data(X, y, seq_len):
    X_b, y_b = [], []
    temp = np.hstack([X[1:], y[:-1]])
    for i in np.random.permutation(temp.shape[0]-seq_len):
        X_b.append(temp[i : i+seq_len, :])
        y_b.append(y[i+1 : i+seq_len+1])
    return np.stack(X_b), np.stack(y_b).reshape((-1, seq_len, 1))

sj = data.loc['sj']
sj_data = processing(sj)
sj_label = label.loc['sj'].values

test_sj = test_data.loc['sj']
test = processing(test_sj)



for length in [20,50,80,100]:

    x_sj, y_sj = batch_data(sj_data[:900], sj_label[:900], length)
    val_x_sj, val_y_sj = batch_data(sj_data[900:], sj_label[900:],1)
    
    for n in [50, 80, 120, 150]:
        model = Sequential()
        model.add(LSTM(n, return_sequences = True, input_shape=(None, x_sj.shape[2])))
        model.add(Dropout(.5))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mae', optimizer=Adam())

        print('seq_num = ' + str(length) + '\t Lstm = ' + str(n))

        history = model.fit(x_sj, y_sj,
                           epochs=3000,
                           batch_size=256,
                           verbose=1,
                           validation_data=[val_x_sj, val_y_sj],
                            callbacks=[EarlyStopping('val_loss', patience=50)])

        loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                             'training': history.history['loss'],
                             'validation': history.history['val_loss']})
        ax = loss.ix[:,:].plot(x='epoch', figsize=(15,8), grid=True)
        fig = ax.get_figure()
        fig.savefig(str(length) + '_' + str(n))



        ans = []
        l = sj_label[-1]
        for i in (test):
            merge = np.hstack([i, l])
            merge = merge.reshape((1,1, merge.shape[0]))
            pred = model.predict(merge)
            l = pred.flatten()
            ans.append(pred)

        np.savetxt(str(length) + '_' + str(n) + '.txt',ans)    
