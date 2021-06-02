""""
Pattern Recongition Course
    Homework 4: Implement cross-validation and grid search

"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score


def kfold(length):
    arr = np.arange(length)
    np.random.shuffle(arr)

def cross_validation(x_train, y_train, k=5):
    k_fold = []
    data_index = kfold(len(y_train))
    for i in range(k):
        k_fold.append([])
    return k_fold

if __name__ == "__main__":
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    
    # Question 1
    kfold_data = cross_validation(x_train, y_train, k=10)
    #assert len(kfold_data) == 10 # should contain 10 fold of data
    #assert len(kfold_data[0]) == 2 # each element should contain train fold and validation fold
    #assert kfold_data[0][1].shape[0] == 55 # The number of data in each validation fold should equal to training data divieded by K
    
    # Question 2
    clf = SVC(C=1.0, kernel='rbf', gamma=0.01)
    
    # Question 3
    
    # Question 4
    