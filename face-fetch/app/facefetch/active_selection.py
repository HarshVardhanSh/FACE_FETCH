import numpy as np

def active(image_representaton, model):

    data = np.array(image_representaton)

    predictions = model.predict_proba(data)

    assert predictions.shape == (data.shape[0], 2)

    reranked_indices  = (predictions[:, :1] - predictions[:, 1:2]).argsort()[::-1][:10]

    return reranked_indices
    
