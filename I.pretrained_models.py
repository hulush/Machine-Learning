import io
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
# Path to the image
img_path = "usb or not.jpg"
# Initialize the model with pre-trained weights
vgg16 = VGG16(weights='imagenet')
# Load image and resize it to 224x224
img = image.load_img(img_path,target_size=(224,224))
array_img = image.img_to_array(img)
array_img = np.expand_dims(array_img, axis=0)
array_img = preprocess_input(array_img)
# Get predictions from our model
predictions = vgg16.predict(array_img)
decoded_predictions = decode_predictions(predictions)
# Decode the predictions
print(str(decoded_predictions))