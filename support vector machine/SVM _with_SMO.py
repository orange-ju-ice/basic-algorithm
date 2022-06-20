import mat4py
import numpy as np
import matplotlib.pyplot as plt

# load dataset
data_mat = mat4py.loadmat(r'xc2.mat')
label = np.array(data_mat.get('c'))
data_x = np.array(data_mat.get('x'))
label_c = np.zeros(label.shape[0])
for z in range(label_c.shape[0]):
    label_c[z] = label[z][0]


# random pick k, k!=j
def selectK(j, p):
    k = j
    while k == j:
        k = np.int(np.random.uniform(0, p))
    return k


def timeBound(beta_c, lamb, sigma, j, k):
    if sigma == 1:
        t_max = min(lamb - beta_c[j], beta_c[k])
        t_min = max(-beta_c[j], beta_c[k] - lamb)
    else:
        t_max = min(lamb - beta_c[j], lamb - beta_c[k])
        t_min = max(-beta_c[j], -beta_c[k])
    return t_min, t_max


def SMO(max_it, tol, lamb, matrix_g, data):
    p = len(data)
    beta = np.zeros([p, max_it + 1])
    delta = 1000
    step = 1
    beta_c = np.zeros(p)
    t = 0
    while step <= max_it and delta > tol:
        for j in range(p):
            k = selectK(j, p)
            sigma = data[j] / data[k]
            vec = np.zeros([p, 1])
            vec[j] = 1
            vec[k] = -sigma
            t_min, t_max = timeBound(beta_c, lamb, sigma, j, k)

            ts = np.dot(vec.T, (np.ones(p).T - np.dot(matrix_g, beta_c)))
            cord = np.dot(vec.T, np.dot(matrix_g, vec))
            tstar = ts / cord

            if cord > 0:
                if t_min > tstar:
                    t = t_min
                elif tstar <= t_max:
                    t = tstar
                else:
                    t = t_max
            elif cord == 0:
                if ts < 0:
                    t = t_min
                elif ts > 0:
                    t = t_max
                else:
                    step = max_it
            else:
                print('error')
                break

            bj_c = beta_c[j] + t
            bk_c = beta_c[k] - sigma * t
            beta_c[j] = bj_c
            beta_c[k] = bk_c

        beta[:, step] = beta_c.copy()
        if step > 1:
            delta = np.linalg.norm(beta[:, step] - beta[:, step - 1]) / np.linalg.norm(beta[:, step - 1])
        step += 1
    return beta[:, step - 1]


def SVM(x_data, c_label):
    max_it = 100
    tol = 0.0001
    lamb = 100
    p = len(c_label)

    data_y = np.zeros(x_data.shape)
    for i in range(p):
        data_y[i] = c_label[i] * x_data[i]
    matrix_g = np.zeros([p, p])
    for i in range(p):
        for j in range(p):
            matrix_g[i, j] = data_y[i].dot(data_y[j])

    beta = SMO(max_it, tol, lamb, matrix_g, c_label)

    logis = np.where(np.logical_and(beta > 0, beta < lamb), True, False)
    data_svx = x_data[logis]
    data_svc = c_label[logis]

    q = np.zeros(2)
    for i in range(p):
        q += beta[i] * data_y[i]

    b = 0
    m = len(data_svx)
    for i in range(m):
        tmp = q.dot(data_svx[i, :]) - data_svc[i]
        b += tmp
    b = b / m
    # b = q.dot(dataSV[2,:]) - data_svc[2]

    return q, b, data_svx


def plotFigure(q, b, x_data, c_label, data_sv):
    cls_1x = x_data[np.where(c_label == 1)]
    cls_2x = x_data[np.where(c_label != 1)]

    plt.scatter(cls_1x[:, 0].flatten(), cls_1x[:, 1].flatten(), s=30, c='r', marker='s')
    plt.scatter(cls_2x[:, 0].flatten(), cls_2x[:, 1].flatten(), s=30, c='purple')

    # 画出 SVM 分类直线
    xx = np.arange(-0.5, 1.15, 0.1)
    # 由分类直线 q[0] * xx + q[1] * yy1 - b = 0 易得下式
    yy1 = (-q[0] * xx + b) / q[1]
    # 由分类直线 q[0] * xx + q[1] * yy2 - b + 1 = 0 易得下式
    yy2 = (-q[0] * xx + b - 1) / q[1]
    # 由分类直线 q[0] * xx + q[1] * yy3 - b - 1 = 0 易得下式
    yy3 = (-q[0] * xx + b + 1) / q[1]
    plt.plot(xx, yy1.T)
    plt.plot(xx, yy2.T)
    plt.plot(xx, yy3.T)

    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.show()


q, b, data_sv = SVM(data_x, label_c)
plotFigure(q, b, data_x, label_c, data_sv)
