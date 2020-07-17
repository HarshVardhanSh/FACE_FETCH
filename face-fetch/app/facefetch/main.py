# Importing all the necessary packages
import numpy as np
from random import shuffle
import os
import shutil
import json
from .model import create_model
from .vgg_features import extract_features, finetune_weights
from .rocchio import find_query_vector
from .anng import knn_ranking
import urllib.parse
import keras
from keras.models import Model
#from active_selection import active
import pandas as pd

dirname = os.path.dirname(__file__)
input_dir = os.path.join(dirname, 'Data')
weights_path = os.path.join(dirname, 'Weights', 'nn4.small2.v1.h5')

images = os.listdir(input_dir)
mymodel = create_model(include_top=False, call = 0)

query_vector = np.zeros((1, 128))

iteration_count = 0
img_to_idx = {}

def init_query_vector():
    '''
    Re Initialize query_vector, mymodel and iteration_count,
    copy over pretrained weights to the current folder.
    '''
    global query_vector
    global mymodel
    global iteration_count
    iteration_count = 0
    mymodel = create_model(include_top = False, call = 0)
    query_vector = np.zeros((1, 128))
    destination_dir = './'
    shutil.copy(weights_path, destination_dir)


import time
def populate_images(similar_images, dissimilar_images, attr=None, act_select = False):
    '''
    Json Object - 3 fields, Similar, Dissimilar if sim/dis are empty then automatically assume flag is false.
    '''
    print("Images Submitted Successfully !")

    global iteration_count
    if iteration_count == 0:
          iteration_count += 1
          if attr:
            face_atr_csv = pd.read_csv("facial_atr.csv")
            global images
            images = list(face_atr_csv[face_atr_csv[attr].apply(lambda x: x == 1).sum(axis = 1) == len(attr)]["image_id"])
    if not(any([similar_images, dissimilar_images])):
        return json.dumps(images[:25 if len(images)>25 else len(images)])
    else:
        image_representation = [{image: 0} for image in images]

        global mymodel
        image_representation = extract_features(image_representation, mymodel)
        keras.backend.clear_session()

        global img_to_idx
        for idx, image_name in enumerate(images):
            img_to_idx[image_name] = idx
        similar_matrix = np.array([image_representation[img_to_idx[image]][image] for image in similar_images])
        dissimilar_matrix = np.array([image_representation[img_to_idx[image]][image] for image in dissimilar_images])

        global query_vector
        query_vector = find_query_vector(similar_matrix, dissimilar_matrix, query_vector, iteration_count)

        if iteration_count == 1:
              num_items = 20
        elif iteration_count == 2:
             num_items = 15
        else:
             num_items = 10
        neighbour_indices = knn_ranking(image_representation, query_vector, num_items)

        labels = [[1, 0] for image in similar_images]
        labels.extend([[0, 1] for image in dissimilar_images])
        labeled_images = similar_images[:]
        labeled_images.extend(dissimilar_images)

        mymodel = create_model(include_top=True, call = None)
        #start = time.time()

        mymodel = finetune_weights(labeled_images, labels, mymodel)

        mymodel = Model(inputs=mymodel.layers[0].output, outputs=mymodel.layers[-2].output)

        if iteration_count == 1:
            images_to_return = [list(image_representation[index].keys())[0] for index in neighbour_indices]
            iteration_count += 1
            return json.dumps(images_to_return)
        print("iteration count ", iteration_count)
        new_image_representation = extract_features([image_representation[index] for index in neighbour_indices], mymodel)

        new_neighbour_indices = knn_ranking(new_image_representation, query_vector, num_items)

        images_to_return = [list(new_image_representation[index].keys())[0] for index in new_neighbour_indices]
        if act_select == True:
            reranked_representation = [list(new_50_representation[index].values())[0] for index in new_neighbour_indices] # where does this variable come from?
            top_ranked = images_to_return[:15]
            low_ranked = images_to_return[15:]
            active_selected_indices = active(reranked_representation[15:], model = mymodel)
            images_to_return = top_ranked.extend([low_ranked[i] for i in active_selected_indices])
            return json.dumps(images_to_return)
        iteration_count += 1
        return json.dumps(images_to_return)
