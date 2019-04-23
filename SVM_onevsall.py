from numpy import linalg
import scipy.io
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from SVM_onevsone import data_matching
from SVM_onevsone import SVM

mat = scipy.io.loadmat('MNIST_data.mat')

train_x = np.array(mat['train_samples'])
y_train = np.array(mat['train_samples_labels']).reshape((mat['train_samples_labels'].shape[0],))

X_test = np.array(mat['test_samples'])
y_test = np.array(mat['test_samples_labels']).reshape((mat['test_samples_labels'].shape[0],))


def oneVsAll(Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9, number):
    if number == 0:
        train_rest = np.vstack((Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9))
        y_train0 = np.ones(len(Train_0))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_0, train_rest, y_train0, y_train_rest

    elif number == 1:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9))
        y_train1 = np.ones(len(Train_1))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_1, train_rest, y_train1, y_train_rest

    elif number == 2:
        train_rest = np.vstack((Train_0, Train_1, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9))
        y_train2 = np.ones(len(Train_2))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_2, train_rest, y_train2, y_train_rest

    elif number == 3:
        train_rest = np.vstack((Train_0, Train_2, Train_1, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9))
        y_train3 = np.ones(len(Train_3))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_3, train_rest, y_train3, y_train_rest

    elif number == 4:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_1, Train_5, Train_6, Train_7, Train_8, Train_9))
        y_train4 = np.ones(len(Train_4))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_4, train_rest, y_train4, y_train_rest

    elif number == 5:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_1, Train_6, Train_7, Train_8, Train_9))
        y_train5 = np.ones(len(Train_5))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_5, train_rest, y_train5, y_train_rest

    elif number == 6:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_5, Train_1, Train_7, Train_8, Train_9))
        y_train6 = np.ones(len(Train_6))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_6, train_rest, y_train6, y_train_rest

    elif number == 7:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_5, Train_6, Train_1, Train_8, Train_9))
        y_train7 = np.ones(len(Train_7))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_7, train_rest, y_train7, y_train_rest

    elif number == 8:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_1, Train_9))
        y_train8 = np.ones(len(Train_8))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_8, train_rest, y_train8, y_train_rest

    elif number == 9:
        train_rest = np.vstack((Train_0, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_1))
        y_train9 = np.ones(len(Train_9))
        y_train_rest = np.ones(len(train_rest)) * -1
        return Train_9, train_rest, y_train9, y_train_rest


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=6):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def main():
    global y_predict
    Train_0, Train_1, Train_2, Train_3, Train_4, Train_5, Train_6, Train_7, Train_8, Train_9 = \
        data_matching(train_x, y_train)

    numpy_predict = []

    for number in range(10):
        train_number, train_rest, test_number, test_rest = oneVsAll(Train_0, Train_1, Train_2, Train_3, Train_4,
                                                                    Train_5, Train_6, Train_7, Train_8, Train_9, number)

        training_data = np.vstack((train_number, train_rest))
        test_data = np.hstack((test_number, test_rest))

        clf = SVM(C=0.1)
        clf.train(training_data, test_data)

        y_predict = clf.compute(X_test)
        numpy_predict.append(y_predict)

    prediction = np.argmax(np.array(numpy_predict), axis=0)

    correct = np.sum(prediction == y_test)

    cnf_matrix = confusion_matrix(y_test, prediction)
    abbreviation = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ax = sns.heatmap(cnf_matrix, cmap=plt.cm.Greens, annot=True, fmt="d")
    ax.set_xticklabels(abbreviation)
    ax.set_yticklabels(abbreviation)
    plt.title('Confusion matrix of SVM_onevsall')
    plt.ylabel('True numbers')
    plt.xlabel('Predicted numbers')
    plt.show()

    size = len(y_predict)
    accuracy = (correct / float(size)) * 100

    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print("The accuracy is  ")
    print(accuracy)


if __name__ == '__main__':
    main()
