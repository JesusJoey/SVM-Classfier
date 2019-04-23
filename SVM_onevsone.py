import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import scipy.io
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from xlwings import xrange

mat = scipy.io.loadmat('MNIST_data.mat')

x_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))

X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))


def data_matching(x_train, y_train):
    Train_0 = []
    Train_1 = []
    Train_2 = []
    Train_3 = []
    Train_4 = []
    Train_5 = []
    Train_6 = []
    Train_7 = []
    Train_8 = []
    Train_9 = []

    for i in xrange(x_train.shape[0]):
        if y_train[i] == 0:
            Train_0.append(x_train[i])
        elif y_train[i] == 1:
            Train_1.append(x_train[i])
        elif y_train[i] == 2:
            Train_2.append(x_train[i])
        elif y_train[i] == 3:
            Train_3.append(x_train[i])
        elif y_train[i] == 4:
            Train_4.append(x_train[i])
        elif y_train[i] == 5:
            Train_5.append(x_train[i])
        elif y_train[i] == 6:
            Train_6.append(x_train[i])
        elif y_train[i] == 7:
            Train_7.append(x_train[i])
        elif y_train[i] == 8:
            Train_8.append(x_train[i])
        elif y_train[i] == 9:
            Train_9.append(x_train[i])

    return np.array(Train_0), np.array(Train_1), np.array(Train_2), np.array(Train_3), np.array(Train_4),\
           np.array(Train_5), np.array(Train_6), np.array(Train_7), np.array(Train_8), np.array(Train_9)
    


def generate_data(X1, X2):
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1

    return y1, y2


def transform_data(predict, plus, minus):
    for i in xrange(predict.shape[0]):
        if predict[i] == 1:
            predict[i] = plus
        elif predict[i] == -1:
            predict[i] = minus

    return predict


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self, kernel=polynomial_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def train(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        Q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, Q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def compute(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.compute(X))


def main():
    Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9 = \
        data_matching(x_train, y_train)

    prediction = []
    np_list = []

    np_predict = [Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9]

    combination = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

    for pair in combination:
        y1, y2 = generate_data(np_predict[pair[0]], np_predict[pair[1]])

        training_data = np.vstack((np_predict[pair[0]], np_predict[pair[1]]))
        test_data = np.hstack((y1, y2))

        clf = SVM(C = 1)
        clf.train(training_data, test_data)

        y_predict = clf.predict(X_test)
        np_list.append(transform_data(y_predict, pair[0], pair[1]))

    np_list = np.array(np_list).astype(int)

    transpose = np.transpose(np_list)

    for row in xrange(transpose.shape[0]):
        counts = np.bincount(transpose[row])
        prediction.append(np.argmax(counts))

    prediction = np.array(prediction)

    correct = np.sum(prediction == y_test)

    cnf_matrix = confusion_matrix(y_test, prediction)
    abbreviation = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ax = sns.heatmap(cnf_matrix, cmap=plt.cm.Greens, annot=True, fmt="d")
    ax.set_xticklabels(abbreviation)
    ax.set_yticklabels(abbreviation)
    plt.title('Confusion matrix of SVM_onevsone')
    plt.ylabel('True numbers')
    plt.xlabel('Predicted numbers')
    plt.show()

    size = len(y_predict)
    accuracy = (correct / float(size)) * 100

    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print("The accuracy is  ")
    print(accuracy, "%")


if __name__ == '__main__':
    main()
