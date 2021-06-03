""""
Pattern Recongition Course
    Homework 4: Implement cross-validation and grid search

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score

C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]


def kfold(length):
    arr = np.arange(length)
    np.random.shuffle(arr)
    return arr


def cross_validation(x_train, y_train, k=5):
    k_fold = []
    n_sample = len(y_train)
    first_fold = n_sample % k
    fold_size = n_sample // k

    index = kfold(n_sample)
    for i in range(k):
        if i < first_fold:
            size = fold_size + 1
            bound = size * i
        else:
            size = fold_size
            bound = first_fold * (fold_size + 1) + (i - first_fold) * size
        validation = index[bound: bound + size]
        training = np.delete(index, np.arange(bound, bound + size))
        print("Split: %s, Training index: %s, Validation index: %s"
              % (i+1, training, validation))
        k_fold.append([training, validation])

    return k_fold


def getFold(x_train, y_train, kfold_data, k):
    train, validate = [], []
    train.append(np.zeros((len(kfold_data[k][0]), len(x_train[0]))))
    train.append(np.zeros((len(kfold_data[k][0]), 1)))
    validate.append(np.zeros((len(kfold_data[k][1]), len(x_train[0]))))
    validate.append(np.zeros((len(kfold_data[k][1]), 1)))

    for i in range(len(kfold_data[k][0])):
        train[0][i] = x_train[kfold_data[k][0][i]]
        train[1][i] = y_train[kfold_data[k][0][i]]
    for i in range(len(kfold_data[k][1])):
        validate[0][i] = x_train[kfold_data[k][1][i]]
        validate[1][i] = y_train[kfold_data[k][1][i]]

    return train, validate


def MSE(test, pred):
    err = 0.0
    for i in range(len(test)):
        err += (test[i] - pred[i]) ** 2
    return err / len(test)


def gridSearch(x_train, y_train, kfold_data, _SVC=True):
    log = np.zeros((len(C), len(gamma)), dtype=np.float32)

    best_c, best_gamma = None, None
    best_acc = 0 if _SVC else -1e8
    for c in range(len(C)):
        for g in range(len(gamma)):
            acc = np.zeros(len(kfold_data))
            for k in range(len(kfold_data)):
                if _SVC:
                    clf = SVC(C=C[c], kernel='rbf', gamma=gamma[g])
                else:
                    clf = SVR(C=C[c], kernel='rbf', gamma=gamma[g])
                train, validate = getFold(x_train, y_train, kfold_data, k)
                clf.fit(train[0], train[1].ravel())
                y_pred = clf.predict(validate[0])
                if _SVC:
                    acc[k] = accuracy_score(validate[1], y_pred)
                else:
                    acc[k] = -1 * MSE(validate[1], y_pred)

            log[c][g] = np.average(acc)
            if log[c][g] > best_acc:
                best_c = C[c]
                best_gamma = gamma[g]
                best_acc = log[c][g]
            log[c][g] = log[c][g] if _SVC else -log[c][g]

    return [best_c, best_gamma], log


def visualize(log):
    fig = plt.figure(figsize=((8, 5)))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(log, cmap="RdBu")
    ax.set_xticks(np.arange(len(gamma)))
    ax.set_yticks(np.arange(len(C)))
    ax.set_xticklabels(gamma)
    ax.set_yticklabels(C)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('C')
    for i in range(len(C)):
        for j in range(len(gamma)):
            ax.text(j, i, f'{log[i, j]:.2f}',
                    ha="center", va="center", color="w")
    ax.set_title("Hyperpparameter Gridsearch")
    ax.figure.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    # Question 1
    kfold_data = cross_validation(x_train, y_train, k=10)
    assert len(kfold_data) == 10
    assert len(kfold_data[0]) == 2
    assert kfold_data[0][1].shape[0] == 55

    # Question 2
    best_para, log_SVC = gridSearch(x_train, y_train, kfold_data)
    print(f"\nBest parameters: C = {best_para[0]}, gamma = {best_para[1]}")

    # Question 3
    visualize(log_SVC)

    # Question 4
    best_model = SVC(C=best_para[0], kernel='rbf', gamma=best_para[1])
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    print("Accuracy score: ", accuracy_score(y_pred, y_test))

    # Question 5
    print("\n-----Question 5----------------------------------")
    train_df = pd.read_csv("../Homework1/train_data.csv")
    x_train = train_df['x_train'].to_numpy().reshape(-1, 1)
    y_train = train_df['y_train'].to_numpy().reshape(-1, 1)

    test_df = pd.read_csv("../Homework1/test_data.csv")
    x_test = test_df['x_test'].to_numpy().reshape(-1, 1)
    y_test = test_df['y_test'].to_numpy().reshape(-1, 1)

    best_para, log_SVR = gridSearch(x_train, y_train, kfold_data, _SVC=False)
    print(f"\nBest parameters: C = {best_para[0]}, gamma = {best_para[1]}")

    visualize(log_SVR)

    best_model = SVR(C=best_para[0], kernel='rbf', gamma=best_para[1])
    best_model.fit(x_train, y_train.ravel())
    y_pred = best_model.predict(x_test)
    mse = MSE(y_test, y_pred)

    print("Square error of Linear regression:     0.4908853488215349")
    print("Square error of SVM regresssion model:", mse[0])
