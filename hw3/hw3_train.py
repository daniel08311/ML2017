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
import numpy,keras,sys
from keras import backend as K
K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")


def create_run_model(X_train,Y_train):

	model = Sequential()
	model.add(Convolution2D(32, (3, 3), activation='relu',input_shape=(1,48,48),padding='same'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.25))
	model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.25))
	model.add(MaxPooling2D((2,2),strides=(2,2)))

	model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.30))
	model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Dropout(0.35))
	model.add(MaxPooling2D((2,2),strides=(2,2))) 


	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu',kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.45))

	model.add(Dense(7, activation='softmax'))
	print (model.summary())
	opt = keras.optimizers.adam(lr=0.00022)
	#opt = keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.000001)
	datagen = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
	datagen.fit(X_train)
	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=256),
	                    steps_per_epoch=110, epochs=350)
	return model

train_file_path = sys.argv[1]

with open(train_file_path,'r') as dest_f:
	data_iter = csv.reader(dest_f, delimiter = ",", quotechar = '"')
	next(data_iter)
	data = [data for data in data_iter]
t_file = numpy.array(data,dtype = str)
train_data = [[pic[0], numpy.array(pic[1].split(' ')).reshape(1,48,48)] for pic in t_file]

X_train, Y_train = [], []
for i in range(len(train_data)):
	X_train.append(train_data[i][1])
	Y_train.append(train_data[i][0])


X_train = numpy.array(X_train,dtype = float)
X_train /= 255
Y_train = numpy.array(Y_train,dtype = float)
Y_train = np_utils.to_categorical(Y_train, 7)

result = create_run_model(X_train,Y_train)
model_json = result.to_json()
with open("model.json", "w") as json_file:
   json_file.write(model_json)
result.save_weights("model.h5") 

