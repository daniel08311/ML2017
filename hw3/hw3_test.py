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
from keras import backend as K
import sys
import csv
import numpy,keras

K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")

test_file_path = sys.argv[1]
output_path = sys.argv[2]

with open(test_file_path,'r') as dest_f:
	data_iter = csv.reader(dest_f, delimiter = ",", quotechar = '"')
	next(data_iter)
	data = [data for data in data_iter]
t_file = numpy.array(data,dtype = str)
train_data = [[pic[0], numpy.array(pic[1].split(' ')).reshape(1,48,48)] for pic in t_file]

X_test = []
for i in range(len(train_data)):
	X_test.append(train_data[i][1])

X_test = numpy.array(X_test,dtype = float)
X_test /= 255

json_file = open('model_0.699.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_0.699.h5")

res = loaded_model.predict_classes(X_test)
csv = open(output_path,"w")
csv.write("id,label\n")
for i in range(len(res)):
	csv.write(str(i) + "," + str(res[i]) + "\n")

