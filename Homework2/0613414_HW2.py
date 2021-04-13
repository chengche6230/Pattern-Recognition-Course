""""
Pattern Recongition Course
    Homework 2: Fisher’s linear discriminant
    
"""
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_train = np.load("x_train.npy") # data point(2D)
    y_train = np.load("y_train.npy") # binary label(0, 1)
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    # 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
    ## code here
    m1, m2 = 0, 0
    
    print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")
    
    # 2. Compute the Within-class scatter matrix SW
    ## code here
    
    assert sw.shape == (2,2)
    print(f"Within-class scatter matrix SW: {sw}")
    
    # 3. Compute the Between-class scatter matrix SB
    ## code here
    
    assert sb.shape == (2,2)
    print(f"Between-class scatter matrix SB: {sb}")
    
    # 4. Compute the Fisher’s linear discriminant
    ## code here
    
    assert w.shape == (2,1)
    print(f" Fisher’s linear discriminant: {w}")
    
    # 5. Project the test data by linear discriminant to get the class prediction 
    #    by nearest-neighbor rule and calculate the accuracy score
    ## code here
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy of test-set {acc}")
    
    # 6. Visualize
    ## code here