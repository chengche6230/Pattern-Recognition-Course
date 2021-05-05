""""
Pattern Recongition Course
    Homework 3: Decision tree and Random forest

"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        return None
    
    def fit(self, X, y):
        return None
    
class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        return None

def gini(sequence):
    gini = 1
    _class = [1, 2]
    for c in _class:
        gini -= (np.sum(sequence==c) / len(sequence))**2
    return gini

def entropy(sequence):
    ent = 0
    _class = [1, 2]
    for i in range(len(_class)):
        p = np.sum(sequence==_class[i]) / len(sequence)
        ent -= p * np.log2(p)
    return ent

if __name__ == "__main__":
    data = load_breast_cancer()
    feature_names = data['feature_names']
    #print(feature_names)
    
    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_test = pd.read_csv("y_test.csv")
    
    # Question 1
    data = np.array([1,2,1,1,1,1,2,2,1,1,2])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))
    
    
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    
    # Visualize
    
    clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
    
    clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])