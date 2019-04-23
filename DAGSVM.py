from numpy import linalg
import scipy.io
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from xlwings import xrange
from SVM_onevsone import data_matching
from SVM_onevsone import SVM

mat = scipy.io.loadmat('MNIST_data.mat')

X_train = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))

X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))


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


def decision_tree(classes, data_values):
    number = classes
    combination = list(itertools.combinations(classes, 2))

    if len(combination) > 1:

        if data_values[combination[0]] < 0:
            number.pop(0)
            return decision_tree(number, data_values)

        else:
            number.pop(1)
            return decision_tree(number, data_values)

    elif len(combination) == 1:

        if data_values[combination[0]] < 0:
            return combination[0][1]

        else:
            return combination[0][0]


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def main():
    X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9 = \
        data_matching(X_train, y_train)

    prediction = []
    np_list = []

    np_predict = [X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9]

    combination = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

    for pair in combination:
        y1, y2 = generate_data(np_predict[pair[0]], np_predict[pair[1]])

        training_data = np.vstack((np_predict[pair[0]], np_predict[pair[1]]))
        test_data = np.hstack((y1, y2))

        clf = SVM(C = 1)
        clf.train(training_data, test_data)

        y_predict = clf.compute(X_test)
        np_list.append(y_predict)

    np_list = np.array(np_list)

    transpose = np.transpose(np_list)

    mix = list(itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

    for row in transpose:

        newdict = {}
        for i in range(len(mix)):
            newdict[mix[i]] = row[i]

        result = decision_tree([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], newdict)
        prediction.append(result)

    cnf_matrix = confusion_matrix(y_test, prediction)
    abbreviation = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ax = sns.heatmap(cnf_matrix, cmap=plt.cm.Greens, annot=True, fmt="d")
    ax.set_xticklabels(abbreviation)
    ax.set_yticklabels(abbreviation)
    plt.title('Confusion matrix of DAGSVM')
    plt.ylabel('True numbers')
    plt.xlabel('Predicted numbers')
    plt.show()

    prediction = np.array(prediction)
    correct = np.sum(prediction == y_test)
    size = len(y_predict)
    accuracy = (correct / float(size)) * 100

    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print("The accuracy is  ")
    print(accuracy)


if __name__ == '__main__':
    main()
