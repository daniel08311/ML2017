from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
import numpy,keras
import matplotlib.pyplot as plt
from PIL import Image

def create_run_model(X_train,Y_train,X_test,Y_test,y_true):

	model = Sequential()
	model.add(Dense(1024, activation='relu',input_dim=2304))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(7, activation='softmax'))
	print (model.summary())
	opt = keras.optimizers.adam(lr=0.00025)

	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	model.fit(X_train, Y_train, validation_data = (X_test, Y_test) ,batch_size=256, nb_epoch=200, verbose=1)
	score =  model.evaluate(X_test,Y_test)
	return model,score[0],score[1] 

X_train = numpy.loadtxt(open("mod/train.csv", "rb"), delimiter=" ")
X_train = X_train.astype('float32')
X_train /= 255

X_test = numpy.loadtxt(open("mod/test.csv", "rb"), delimiter=" ")
X_test = X_test.astype('float32')
X_test /= 255

Y_train = numpy.loadtxt(open("mod/Y_train.csv", "rb"), delimiter=",")
y_true = Y_train

n_folds = 10
skf = StratifiedKFold(Y_train, n_folds=n_folds, shuffle=True)

Y_train = np_utils.to_categorical(Y_train, 7)

for i, (train, test) in enumerate(skf):
	x_train = X_train[train]
	x_test = X_train[test]
	y_train = Y_train[train]
	y_test = Y_train[test]
	print ("\t\nRunning Fold", i+1, "/", n_folds)
	result = create_run_model(x_train,y_train,x_test,y_test,y_true)
	print (result[1],result[2])

res = result[0].predict_classes(X_test)
res.save("model.h5")

