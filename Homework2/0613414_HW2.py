""""
Pattern Recongition Course
    Homework 2: Fisher’s linear discriminant

"""
import numpy as np
from numpy import matmul as mul
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt


def findNearest(y, y_train, w, test):
    _min = 1e8
    pre_label = 0
    y_pre = mul(w.T, test)  # continous value
    for i in range(len(y)):
        if abs(y_pre - y[i]) < _min:
            _min = abs(y_pre - y[i])
            pre_label = y_train[i]
    return pre_label

if __name__ == "__main__":
    x_train = np.load("x_train.npy")  # data point(2D)
    y_train = np.load("y_train.npy")  # binary label(0, 1)
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    data_dim = x_train.shape[1]

    # 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
    m = np.array([np.zeros((data_dim)), np.zeros((data_dim))])
    for i in range(len(y_train)):
        for d in range(data_dim):
            m[y_train[i]][d] += x_train[i][d]
    for c in range(2):
        m[c] /= np.sum(y_train == 0 + c)
    print(f"Mean vector of class 1: {m[0]}",
          f"\nMean vector of class 2: {m[1]}\n")

    # 2. Compute the Within-class scatter matrix SW
    sw = np.zeros((2, 2))
    for i in range(len(y_train)):
        C = y_train[i]
        t_sw = (x_train[i] - m[C]).reshape(data_dim, 1)
        sw += mul(t_sw, t_sw.T)
    assert sw.shape == (2, 2)
    print(f"Within-class scatter matrix SW:\n{sw}\n")

    # 3. Compute the Between-class scatter matrix SB
    t_sb = (m[1] - m[0]).reshape(data_dim, 1)
    sb = mul(t_sb, t_sb.T)
    assert sb.shape == (2, 2)
    print(f"Between-class scatter matrix SB:\n{sb}\n")

    # 4. Compute the Fisher’s linear discriminant
    w = mul(np.linalg.inv(sw), t_sb)
    assert w.shape == (2, 1)
    print(f" Fisher’s linear discriminant:\n{w}\n")

    # 5. Project the test data and calculate the accuracy score

    # Project data
    y = np.zeros((len(y_train)))
    for i in range(len(y_train)):
        y[i] = mul(w.T, x_train[i])

    # Predict data
    y_pred = np.zeros((len(y_test)), np.uint8)
    for i in range(len(y_test)):
        y_pred[i] = findNearest(y, y_train, w, x_test[i])

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy of test-set {acc}")

    # 6. Visualize
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    slope = float(w[1] / w[0])
    intercep = 0
    title = "Projection line: w = " + str(slope) + ", b = " + str(intercep)
    colors = ['tab:red', 'tab:blue']
    ax.set_title(title)

    # Test data points
    ax.scatter(x_test[:, 0], x_test[:, 1],
               c=y_test, cmap=matplotlib.colors.ListedColormap(colors))

    # Project points
    colors_proj = ['red', 'blue']
    x_proj = np.zeros((len(y_test)))
    y_proj = np.zeros((len(y_test)))
    for i in range(len(y_test)):
        x_proj[i] = (w[0]/w[1]) * x_test[i][0] + x_test[i][1]
        x_proj[i] /= (w[1]/w[0] + w[0]/w[1])
        y_proj[i] = (w[1] / w[0]) * x_proj[i]
        ax.plot([x_test[i][0], x_proj[i]], [x_test[i][1], y_proj[i]],
                color="tab:gray", lw=0.5)
    ax.scatter(x_proj, y_proj,
               c=y_test, cmap=matplotlib.colors.ListedColormap(colors_proj))

    # Projection line
    X = np.linspace(min(x_proj), max(x_proj), 200)
    Y = np.zeros((200))
    for i in range(len(X)):
        Y[i] = X[i] * slope + intercep
    ax.plot(X, Y, "k")

    plt.show()
