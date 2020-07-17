# Importing all the necessary packages
import numpy as np
import pandas as pd
import os
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input


# Specifying the paths for the input directory and the model
input_dir = '/Users/ramapinnimty/Desktop/CBIR/workspace/Data'
model_path = '/Users/ramapinnimty/Desktop/CBIR/workspace/vgg_face_weights.h5'

def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

def extract_features(image_representation, model = None):
        feature_matrix = []
        for image in image_representation:
                img = preprocess_image(os.path.join(input_dir, image.keys()[0]))
                _ = model.predict(img)
                features = model.layers[-5].reshape(4096)
                feature_matrix.append(features)
        return feature_matrix

def finetune_weights(images, labels, model = None):
        data = []
        for image in images:
                img = preprocess_image(os.path.join(input_dir, image))
                data.append(img)
        data = np.array(data)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(data, labels, epochs=10, batch_size=1)
        return model



