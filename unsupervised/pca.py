#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import syntheticdata
import random


def center_data(A):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # X    [NxM] numpy centered data matrix (N samples, M features)
    A = A -np.mean(A,axis=0)
    return A


def test_center():
    testcase = np.array([[3., 11., 4.3], [4., 5., 4.3], [5., 17., 4.5], [4, 13., 4.4]])
    answer = np.array([[-1., -0.5, -0.075], [0., -6.5, -0.075], [1., 5.5, 0.125], [0., 1.5, 0.025]])
    np.testing.assert_array_almost_equal(center_data(testcase), answer)
    print(answer)
    print("-----")
    print(center_data(testcase))


def compute_covariance_matrix(A):
    # INPUT:
    # A    [NxM] centered numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # C    [MxM] numpy covariance matrix (M features, M features)
    #
    # Do not apply centering here. We assume that A is centered before this function is called.
    n = A.shape[0]
    C = 1 / n * A.T @ A
    return C


def test_matrix():
    testcase = center_data(np.array([[22., 11., 5.5], [10., 5., 2.5], [34., 17., 8.5], [28., 14., 7]]))
    answer = np.array([[580., 290., 145.], [290., 145., 72.5], [145., 72.5, 36.25]])
    # Depending on implementation the scale can be different:
    to_test = compute_covariance_matrix(testcase)
    answer = answer / answer[0, 0]
    to_test = to_test / to_test[0, 0]
    np.testing.assert_array_almost_equal(to_test, answer)
    print(answer)
    print("----")
    print(to_test)


def compute_eigenvalue_eigenvectors(A):
    # INPUT:
    # A    [DxD] numpy matrix
    #
    # OUTPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors

    eigval, eigvec = np.linalg.eig(A)  # None, None

    # Numerical roundoff can lead to (tiny) imaginary parts. We correct that here.
    eigval = eigval.real
    eigvec = eigvec.real

    return eigval, eigvec


def test_eigen():
    testcase = np.array([[2, 0, 0], [0, 5, 0], [0, 0, 3]])
    answer1 = np.array([2., 5., 3.])
    answer2 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    x, y = compute_eigenvalue_eigenvectors(testcase)
    np.testing.assert_array_almost_equal(x, answer1)
    np.testing.assert_array_almost_equal(y, answer2)


def sort_eigenvalue_eigenvectors(eigval, eigvec):
    # INPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors
    #
    # OUTPUT:
    # sorted_eigval    [D] numpy vector of eigenvalues
    # sorted_eigvec    [DxD] numpy array of eigenvectors
    """inpsirert av:
    https://python-decompiler.com/article/2011-11/sort-eigenvalues-and-associated-eigenvectors-after-
    using-numpy-linalg-eig-in-pyt"""
    idx = eigval.argsort()[::-1]
    sorted_eigval = eigval[idx]
    sorted_eigvec = eigvec[:,idx]

    return sorted_eigval, sorted_eigvec


def test_sort_eigen():
    testcase = np.array([[2, 0, 0], [0, 5, 0], [0, 0, 3]])
    answer1 = np.array([5., 3., 2.])
    answer2 = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    x, y = compute_eigenvalue_eigenvectors(testcase)
    x, y = sort_eigenvalue_eigenvectors(x, y)
    np.testing.assert_array_almost_equal(x, answer1)
    np.testing.assert_array_almost_equal(y, answer2)


def pca(A, m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of learned features (m <= M)
    #
    # OUTPUT:
    # pca_eigvec    [mxM] numpy matrix containing the eigenvectors (m eigenvectors, M dimensions)
    # P    [Nxm] numpy PCA data matrix (N samples, m features)
    A = center_data(A)
    C = compute_covariance_matrix(A)
    eigVal, eigVec = compute_eigenvalue_eigenvectors(C)
    eigVal, eigVec = sort_eigenvalue_eigenvectors(eigVal, eigVec)
    pca_eigvec = eigVec[:, :m]
    P = A@pca_eigvec
    return pca_eigvec, P


def test_pca():
    testcase = np.array([[22., 11., 5.5], [10., 5., 2.5], [34., 17., 8.5]])
    x, y = pca(testcase, 2)
    import pickle
    answer1_file = open('PCAanswer1.pkl', 'rb');
    answer2_file = open('PCAanswer2.pkl', 'rb')
    answer1 = pickle.load(answer1_file);
    answer2 = pickle.load(answer2_file)

    test_arr_x = np.sum(np.abs(np.abs(x) - np.abs(answer1)), axis=0)
    np.testing.assert_array_almost_equal(test_arr_x, np.zeros(2))

    test_arr_y = np.sum(np.abs(np.abs(y) - np.abs(answer2)))
    np.testing.assert_almost_equal(test_arr_y, 0)


def plot_data():
    X = syntheticdata.get_synthetic_data1()
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    X = center_data(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def pca_on_iris():
    X, y = syntheticdata.get_iris_data()
    idx = random.sample(range(0, X.shape[1]), 2)
    print(idx)
    data = X[:, idx]
    plt.scatter(data[:, 0], data[:, 1], c=y)
    plt.figure()
    _, P = pca(data, 2)
    plt.scatter(P[:, 0], P[:, 1], c=y)
    plt.show()


def plot_pca():
    X = syntheticdata.get_synthetic_data1()
    X = center_data(X)
    pca_eigvec, _ = pca(X, 2)
    first_eigvec = pca_eigvec[0, :]

    plt.scatter(X[:, 0], X[:, 1])

    x = np.linspace(-5, 5, 1000)
    y = first_eigvec[1] / first_eigvec[0] * x
    plt.plot(x, y)
    plt.show()
    _, P = pca(X, 2)
    P[:, 1] = 0
    # P = P@pca_eigvec.T
    plt.scatter(P[:, 0], P[:, 1])
    plt.show()


def pca_on_labeled_data():
    X = syntheticdata.get_synthetic_data1()
    _, P = pca(X, 2)
    X, y = syntheticdata.get_synthetic_data_with_labels1()
    P[:, 1] = 0
    plt.scatter(P[:, 0], P[:, 1], c=y[:, 0])

    plt.figure()
    _, P = pca(X, 2)
    # P[:,1] = 1
    # P = P@pca_eigvec.T
    plt.scatter(P[:, 0], np.ones(P.shape[0]), c=y[:, 0])
    # plt.scatter(P[:,0],P[:,1],c=y[:,0])
    plt.show()


def pca_on_labeled_data2():
    X, y = syntheticdata.get_synthetic_data_with_labels2()
    plt.scatter(X[:, 0], np.zeros(X.shape[0]), c=y[:, 0])
    plt.figure()
    _, P = pca(X, 2)
    plt.scatter(P[:, 0], np.zeros(P.shape[0]), c=y[:, 0])
    plt.show()
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0])
    plt.figure()
    _, P = pca(X, 2)
    plt.scatter(P[:, 0], P[:, 1], c=y[:, 0])
    plt.show()


def encode_decode_pca(A, m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of learned features (m <= M)
    #
    # OUTPUT:
    # Ahat [NxM] numpy PCA reconstructed data matrix (N samples, M features)
    eigVec, Z = pca(A, m)
    Ahat = Z @ eigVec.T
    return Ahat


def test_encode_decode():
    X, y, h, w = syntheticdata.get_lfw_data()
    plt.imshow(X[0, :].reshape((h, w)), cmap=plt.cm.gray)
    plt.show()

def encode_decode_pca_with_pov(A,p):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # p    float number between 0 and 1 denoting the POV to be preserved
    # OUTPUT:
    # Ahat [NxM] numpy PCA reconstructed data matrix (N samples, M features)
    # m    integer reporting the number of dimensions selected
    A2 = center_data(A)
    C = compute_covariance_matrix(A2)
    eigVal, eigVec = compute_eigenvalue_eigenvectors(C)
    eigVal, eigVec = sort_eigenvalue_eigenvectors(eigVal, eigVec)
    sum_eigVal = sum(eigVal)
    pov = sum_eigVal*p
    print(pov)
    the_sum = 0
    teller = 0
    m = 0
    while the_sum<pov:
        the_sum += eigVal[m]
        print(the_sum)
        m+=1
    cov_sum = sum([C[x][x] for x in range(C.shape[0])])
    print(cov_sum,sum_eigVal)
    print(p,(the_sum/sum_eigVal))
    eigVec, Z = pca(A, m)
    Ahat = Z @ eigVec.T
    return Ahat, m


def test_encode_decode_pov():
    X, y, h, w = syntheticdata.get_lfw_data()
    Xhat,m = encode_decode_pca_with_pov(X,0.7)
    print(m)
    plt.imshow(Xhat[0, :].reshape((h, w)), cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    """test_center()
    test_matrix()
    test_eigen()
    test_sort_eigen()
    test_pca()
    plot_data()
    plot_pca()
    pca_on_iris()
    pca_on_labeled_data()
    pca_on_labeled_data2()
    test_encode_decode()"""
    test_encode_decode_pov()
