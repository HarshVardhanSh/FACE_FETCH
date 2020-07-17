import numpy as np
def find_query_vector(relevant_matrix, irrelevant_matrix, query_vector):
    alpha = 1.0
    beta = 0.8
    gamma = 0.1

    term1 = alpha * query_vector
    term2 = beta * (1/relevant_matrix.shape[0]) * np.sum(relevant_matrix, axis = 0)
    term3 = gamma * (1/irrelevant_matrix.shape[0]) * np.sum(irrelevant_matrix, axis = 0)

    query_vec = term1 + term2 - term3

    return query_vec
