import numpy as np
import matplotlib.pyplot as plt


def build_normal_matrix(m, n):
    np.random.seed(200)
    matrix_a = np.random.normal(0, 1, (m, n))
    matrix_a_norm = np.linalg.norm(matrix_a, axis=1, ord=2)
    return matrix_a, np.array(matrix_a_norm).reshape(m, 1)


def classical_K(A, b, x_acc, x0, iter_max):
    x = x0
    res = x - x_acc
    res_norm_record = [np.linalg.norm(res, 2)]

    for k in range(iter_max):
        r = k % m
        a = A[r].reshape((1, n))
        v = b[r] - np.dot(a, x)
        v_norm = np.linalg.norm(a, 2) ** 2
        v = v / v_norm
        x = x + v * a.T
        r_norm2 = np.linalg.norm(x - x_acc, 2)
        res_norm_record.append(float(r_norm2))

    return res_norm_record


def RK_delete(matrix_a, vector_b, x_round):
    # matrix_a=np.array(matrix_a)
    # vector_b=np.array(vector_b)
    # x_round=np.array(x_round)
    error = abs(vector_b - np.dot(matrix_a, x_round))
    row_del = np.argmax(error)
    matrix_a = np.delete(matrix_a, row_del, axis=0)
    vector_b = np.delete(vector_b, row_del, axis=0)
    return matrix_a, vector_b


def RK_Removal(matrix_a, vector_b, x_acc, x0, iter_max, matrix_a_norm):
    x = x0
    res = x - x_acc
    res_norm_record = [np.linalg.norm(res, 2)]

    matrix_a_norm = matrix_a_norm ** 2 / np.sum(matrix_a_norm ** 2)
    A_csum = np.cumsum(matrix_a_norm)

    np.random.seed()
    for k in range(iter_max):
        rand_num = np.random.uniform(0, 1)
        r = np.min(np.where(A_csum > rand_num))
        a = matrix_a[r].reshape((1, n))
        v = vector_b[r] - np.dot(a, x)
        v_norm = np.linalg.norm(a, 2) ** 2
        v = v / v_norm
        x = x + v * a.T
        r_norm2 = np.linalg.norm(x - x_acc, 2)
        res_norm_record.append(float(r_norm2))
        if k % 800 == 0:
            matrix_a, vector_b = RK_delete(matrix_a, vector_b, np.array(x).reshape((n, 1)))
            matrix_a_norm = np.linalg.norm(matrix_a, axis=1, ord=2)
            matrix_a_norm = matrix_a_norm ** 2 / np.sum(matrix_a_norm ** 2)
            A_csum = np.cumsum(matrix_a_norm)
    return res_norm_record


def upper_bound(A, x, x_acc, iter_max, A_norm):
    _, sigma, _ = np.linalg.svd(A)
    r0_norm = np.linalg.norm(x - x_acc)
    alpha = 1 - sigma[-1] ** 2 / np.sum(A_norm ** 2)
    k = np.arange(0, iter_max + 1)
    r_norm_bd = r0_norm * alpha ** (k / 2)

    return r_norm_bd


if __name__ == '__main__':

    iter_max = 10000
    run_max = 20
    figcont = 0
    cont = np.arange(0, iter_max + 1)

    m = 800
    n = 41

    A_nm, A_nm_norm = build_normal_matrix(m, n)
    x_true = np.ones((n, 1))
    x = np.random.uniform(-100, 100, (n, 1))
    b_star = np.dot(A_nm, x_true)
    b_c = np.zeros((m, 1), dtype='float')
    for i in range(m):
        if i % 150 == 0:
            b_c[i] = 100 * np.random.random()
    print(b_c)
    b = b_star + b_c
    r_norm_RK = []

    for i in range(run_max):
        r_norm_RK.append(RK_Removal(A_nm, b, x_true, x, iter_max, A_nm_norm))
        if i % 5 == 0:
            print(str(i))
    # r_norm_RK.append(RK(A_nm, b, x_true, x, iter_max, A_nm_norm))
    # print(np.array(r_norm_RK).shape)
    r_norm_RK = np.mean(np.array(r_norm_RK), axis=0)
    # print(r_norm_RK.shape)

    # Plot Figure
    figcont += 1
    plt.figure(figcont, figsize=(9, 7))
    plt.semilogy(cont, r_norm_RK,
                 linewidth='4', linestyle='-',
                 color='mediumorchid', label="RK Removal")

    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.xticks(np.arange(0, 10) * 1000)
    plt.yticks(np.logspace(-16, 4, num=5))
    plt.xlim(0, iter_max)
    plt.ylim(1e-16, 1e+4)
    legend = plt.legend(fontsize=17)
    plt.ylabel(r"$E||x_{k}-x||_2 $", fontsize="22")
    plt.xlabel("Number of projections " + r"$k$", fontsize="22")
    plt.tick_params(labelsize=15)
    plt.title("Gaussian " + str(m) + " by " + str(n), fontsize=24)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    # plt.savefig("fig2_basic_Gaussian.jpg")
    plt.show()
