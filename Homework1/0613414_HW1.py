""""
Pattern Recongition Courese
Homework 1

Requirment:
    1.Use MAE and MSE as the loss function to do linear regression.
    2.Calculate loss and weights.
    3.Plot the learning curve.
    4.Use numpy only and follow PEP8 coding style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def model(x, b0, b1):
    return x * b1 + b0


def MAE(pre, y):
    return np.average(abs(pre - y))


def MSE(pre, y):
    return np.average((pre - y)**2)


def MSE_main(
        b0, b1, lr, batch_size, batch_num,
        converge_count, converge_thres,
        x_train, y_train):
    log = {'b0': [], 'b1': [], 'loss': []}
    count = 0

    while count < converge_count:  # loop until convergence

        # Predict and Criterion
        py = model(x_train, b0, b1)
        loss = MSE(py, y_train)
        if len(log['loss']) > 0:
            if abs(loss - log['loss'][-1]) < converge_thres:
                count += 1
            else:
                count = 0

        log['b0'].append(b0)
        log['b1'].append(b1)
        log['loss'].append(loss)

        # Gradient descent and learn
        batch = np.random.randint(batch_num)  # randomly pick a batch
        da, db = 0.0, 0.0  # a:b1, b:b0
        for k in range(batch*batch_size, (batch+1)*batch_size):
            da += (model(x_train[k], b0, b1) - y_train[k]) * x_train[k]
            db += model(x_train[k], b0, b1) - y_train[k]
        da = da/batch_size
        db = db/batch_size

        b0 = b0 - lr * db
        b1 = b1 - lr * da

    return log, b0, b1


def MAE_main(
        b0, b1, lr, batch_size, batch_num,
        converge_count, converge_thres,
        x_train, y_train):
    log = {'b0': [], 'b1': [], 'loss': []}
    count = 0

    while count < converge_count:  # loop until convergence

        # Predict and Criterion
        py = model(x_train, b0, b1)
        loss = MAE(py, y_train)
        if len(log['loss']) > 0:
            if abs(loss - log['loss'][-1]) < converge_thres:
                count += 1
            else:
                count = 0

        log['b0'].append(b0)
        log['b1'].append(b1)
        log['loss'].append(loss)

        # Gradient descent and learn
        batch = np.random.randint(batch_num)  # randomly pick a batch
        da, db = 0.0, 0.0  # a:b1, b:b0
        for k in range(batch*batch_size, (batch+1)*batch_size):
            if model(x_train[k], b0, b1) >= y_train[k]:
                da += x_train[k]
                db += 1
            else:
                da -= x_train[k]
                db -= 1
        da = da/batch_size
        db = db/batch_size

        b0 = b0 - lr * db
        b1 = b1 - lr * da

    return log, b0, b1


if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv('train_data.csv')
    x_train, y_train = train_df['x_train'], train_df['y_train']

    # Train model
    lr = 1e-2
    batch_size = 50
    batch_num = len(x_train)//batch_size
    converge_count = 20
    converge_thres = 1e-5
    b0 = np.random.normal()
    b1 = np.random.normal()

    mse_log, mse_b0, mse_b1 = MSE_main(b0, b1, lr, batch_size, batch_num,
                                       converge_count, converge_thres,
                                       x_train, y_train)
    mae_log, mae_b0, mae_b1 = MAE_main(b0, b1, lr, batch_size, batch_num,
                                       converge_count, converge_thres,
                                       x_train, y_train)

    # Test model
    test_data = pd.read_csv("test_data.csv")
    x_test, y_test = test_data['x_test'], test_data['y_test']
    mse_y_pred = model(x_test, mse_b0, mse_b1)
    mae_y_pred = model(x_test, mae_b0, mae_b1)
    mse_loss = MSE(mse_y_pred, y_test)
    mae_loss = MAE(mae_y_pred, y_test)

    # Output result
    print("MSE (run %d times)" % (len(mse_log['loss'])))
    print("\tloss:", mse_loss)
    print("\tb0: %.10f, b1:%.10f" % (mse_b0, mse_b1))
    print("MAE (run %d times)" % (len(mae_log['loss'])))
    print("\tloss:", mae_loss)
    print("\tb0: %.10f, b1:%.10f" % (mae_b0, mae_b1))

    # Visualize
    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.plot(x_test, y_test, '.k')
    ax.plot(x_test, mse_b1 * x_test + mse_b0, 'r', label='MSE line')
    ax.plot(x_test, mae_b1 * x_test + mae_b0, 'g', label='MAE line')
    plt.legend(loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    ax.plot(mse_log['loss'], 'r', label='MSE loss')
    ax.plot(mae_log['loss'], 'g', label='MAE loss')
    plt.legend()
