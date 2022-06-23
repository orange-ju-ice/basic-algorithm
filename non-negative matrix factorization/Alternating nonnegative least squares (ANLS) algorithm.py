import numpy as np


def cal_L_inverse(w_matrix):
    column = w_matrix.shape[1]
    l_matrix = np.zeros((column, column))
    for c in range(column):
        l_matrix[c, c] = 1 / max(w_matrix[:, c])
    return l_matrix


def nmf(v_matrix, k, max_it, err=0.000001):
    rows, columns = v_matrix.shape
    t = 0
    w = np.random.random((rows, k))
    h = np.random.random((k, columns))
    for i in range(max_it):
        v_c = np.dot(w, h)
        err_matrix = v_matrix - v_c
        print("error=%s,iter=%d" % (np.sum(err_matrix ** 2), t))
        h_new = np.maximum(0, np.dot(w.T, v_matrix) / (np.dot(w.T, v_c)) * h)
        v_c = np.dot(w, h_new)
        w_new = np.maximum(0, (np.dot(v_matrix, h_new.T)) / (np.dot(v_c, h_new.T)) * w)
        w_new = np.dot(w_new, cal_L_inverse(w_new))
        delta = np.sqrt(np.sum((w_new - w) ** 2) / np.sum(w ** 2)) + np.sqrt(np.sum((h_new - h) ** 2) / np.sum(h * h))
        w, h = w_new, h_new
        t = t + 1
        if delta < err:
            break
    return w, h


x = np.random.random((64, 32))
nmf(x, 16, 8000)
