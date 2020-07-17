import ngtpy
import csv
import tensorflow as tf
import numpy as np
# create an index framwork in filesystem.
def knn_ranking(image_representation, query_vector, num_items):
    ngtpy.create(path='index', dimension=128, distance_type="L2")

    objects = []
    for image in image_representation:
        tensor_temp = list(image.values())[0]
        objects.append(tensor_temp)

    objects = np.array(objects).reshape(len(image_representation),128)
#     print(np.array(objects).shape)
    index = ngtpy.Index('index')
    index.batch_insert(objects)


    index.save()
    index.close()


    index = ngtpy.Index(b'index')

    results = index.search(query_vector, size=num_items)

    results_main = [i for (i,j) in results[:num_items]]
    print(results_main)

    return results_main
