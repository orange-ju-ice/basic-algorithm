import numpy as np


def pca(data_matrix, k):
    samples, features = data_matrix.shape
    mean = np.array([np.mean(data_matrix[:, i]) for i in range(features)])
    # normalization
    norm_data = data_matrix - mean
    cov_matrix = np.dot(norm_data.T, norm_data)
    # calculate eigenvector and eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov_matrix)
    eig_pairs = [(eig_val[i], eig_vec[:, i]) for i in range(features)]
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    p_matrix = np.array([ele[1] for ele in eig_pairs[:k]])
    new_data = np.dot(norm_data, p_matrix.T)
    return new_data


X = np.array([[-1, 1, 1], [-2, -1, 1], [-3, -2, 1], [1, 1, 1], [2, 1, 1], [3, 2, 1]])
print(pca(X, 1))
