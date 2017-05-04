import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import model_from_json
from keras.models import load_model, Sequential, Model
from vis.utils import utils
from vis.visualization import visualize_saliency
from PIL import Image
from keras import backend as K
K.set_image_data_format("channels_first")
K.set_image_dim_ordering("th")

json_file = open('model_0.699.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_0.699.h5")

img_idx = [7,99,136,213,229]
data = pd.read_csv("train_raw.csv",nrows = np.max(img_idx)+1)
x = np.array([ele[1].split() for ele in data.values],dtype = float).reshape(data.shape[0],1,48,48)
imgs = x[img_idx]
pred_class = loaded_model.predict_classes(imgs/255)

heatmaps = []
for img , p_class in zip(imgs,pred_class):
    seed = img.reshape(48,48,1)
    heatmap = visualize_saliency(loaded_model, len(loaded_model.layers)-1, [p_class], seed, alpha=0.5)
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()
