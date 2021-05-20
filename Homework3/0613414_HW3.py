""""
Pattern Recongition Course
    Homework 3: Decision tree and Random forest

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

_class = [0, 1]
train_num = 426
test_num = 143
feature_num = 30
feature_name = None

def gini(sequence):
    gini = 1
    for i in range(len(_class)):
        gini -= (np.sum(sequence==_class[i]) / len(sequence))**2
    return gini

def entropy(sequence):
    ent = 0
    for i in range(len(_class)):
        p = np.sum(sequence==_class[i]) / len(sequence)
        if p!=0:
            ent -= p * np.log2(p)
    return ent

def impurity(sequence, mode):
    if mode == 'gini':
        return gini(sequence)
    elif mode == 'entropy':
        return entropy(sequence)
    else:
        return None

class Tree():
    def __init__(self, x, depth, isLeaf=False):
        self.x = x
        self.depth = depth
        self.isLeaf = isLeaf
        self.info = 0
        self.left = None
        self.right = None
        self.attr_index = -1
        self.threshold = -1
        
    def setting(self, attr_index, threshold, left, right):
        self.left = left
        self.right = right
        self.attr_index = attr_index
        self.threshold = threshold
    
    def Print(self):
        print('----------------------------')
        print(f'#:{len(self.x)} Depth:{self.depth}, isLeaf:{self.isLeaf}, info:{self.info}')
        print(f"0:{np.sum(self.x['label']==0)}, 1:{np.sum(self.x['label']==1)}")
   
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.importance = dict.fromkeys(feature_name, 0)
    
    def splitAttr(self, x, info):
        attr_index, threshold = 0, 0.0
        info_gain = 0
        for col in range(len(x.columns)-1): # exclude label column
            tmp_x = x.sort_values(by=[x.columns[col]])
            for i in range(len(x)):
                left = impurity(tmp_x.values[0:i+1, -1], self.criterion)
                right = impurity(tmp_x.values[i:train_num, -1], self.criterion)
                info_p = ((i + 1) * left + (train_num - i + 1) * right) / train_num
                if info - info_p > info_gain:
                    info_gain = info - info_p
                    attr_index, threshold = col, tmp_x.values[i, col]
        return info_gain, attr_index, threshold
    
    def partiTree(self, tree, attr_index, threshold):
        left_df = pd.DataFrame(columns=tree.x.columns)
        right_df = pd.DataFrame(columns=tree.x.columns)
        for i in range(len(tree.x)):
            if tree.x.values[i, attr_index] <= threshold:
                left_df = left_df.append(tree.x.loc[tree.x.index[i]])
            else:
                right_df = right_df.append(tree.x.loc[tree.x.index[i]])
        left = Tree(left_df, tree.depth + 1)
        right = Tree(right_df, tree.depth + 1)
        if len(left.x) <= 0:
            left.isLeaf = True
        if len(right.x) <= 0:
            right.isLeaf = True
        return left, right
        
    def genTree(self, tree):
        # Init tree info.
        tree.info = impurity(tree.x['label'], self.criterion)
        if tree.depth >= self.max_depth or tree.info == 0:
            tree.isLeaf = True
            return tree
        
        for _iter in range(10):
            # Select best split attribute and threshold
            info_gain, attr_index, threshold = self.splitAttr(tree.x, tree.info)
            if info_gain == 0:
                tree.isLeaf = True
                return tree
            
            # Partition tree
            left, right = self.partiTree(tree, attr_index, threshold)
            if len(left.x) != len(tree.x) and len(right.x) != len(tree.x):
                break
        tree.setting(attr_index, threshold, left, right)
        
        # Move to tree's children
        if not left.isLeaf:
            self.genTree(left)
        if not right.isLeaf:
            self.genTree(right)
    
    def fit(self, x):
        self.tree = Tree(x, 1)
        self.genTree(self.tree)
        
    def model(self, test):
        # Given X, return a predict class
        tmp = self.tree
        while True:
            if test[tmp.attr_index] <= tmp.threshold:
                if tmp.left == None:
                    break
                tmp = tmp.left
            else:
                if tmp.right == None:
                    break
                tmp = tmp.right
        pre = np.zeros((len(_class)), dtype=np.uint16)
        for c in range(len(_class)):
            pre[c] = np.sum(tmp.x['label']==_class[c])    
        return _class[np.argmax(pre)]
        
    def predict(self, X):
        pred_y = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            pred_y[i] = self.model(X.values[i])  
        return pred_y
    
    def countImportance(self, tree):
        if tree.attr_index != -1:
            self.importance[feature_name[tree.attr_index]] += 1
        
        if tree.left != None:
            self.countImportance(tree.left)
        if tree.right != None:
            self.countImportance(tree.right)

def visualize(importance):
    imp = dict(sorted(importance.items(), key=lambda d: d[1]))
    imp = {k:v for k,v in imp.items() if v != 0} # rm empty value
    plt.style.use('ggplot')
    plt.title('Feature Importance')
    plt.barh(list(imp.keys()),imp.values())
    plt.show()
    
class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = int(max_features)
        self.boostrap = boostrap
        self.criterion = criterion
        self.max_depth = 15
        self.forest = []
    
    def splitAttr(self, x, info):
        attr_index, threshold = 0, 0.0
        info_gain = 0
        for col in range(len(x.columns)-1): # exclude label column
            tmp_x = x.sort_values(by=[x.columns[col]])
            for i in range(len(x)):
                left = impurity(tmp_x.values[0:i+1, -1], self.criterion)
                right = impurity(tmp_x.values[i:train_num, -1], self.criterion)
                info_p = ((i + 1) * left + (train_num - i + 1) * right) / train_num
                if info - info_p > info_gain:
                    info_gain = info - info_p
                    attr_index, threshold = col, tmp_x.values[i, col]
        attr_index = feature_name.index(x.columns[attr_index])
        return info_gain, attr_index, threshold
    def partiTree(self, tree, attr_index, threshold):
        left_df = pd.DataFrame(columns=tree.x.columns)
        right_df = pd.DataFrame(columns=tree.x.columns)
        for i in range(len(tree.x)):
            if tree.x.values[i, attr_index] <= threshold:
                left_df = left_df.append(tree.x.iloc[i])
            else:
                right_df = right_df.append(tree.x.iloc[i])
        left = Tree(left_df, tree.depth + 1)
        right = Tree(right_df, tree.depth + 1)
        if len(left.x) <= 0:
            left.isLeaf = True
        if len(right.x) <= 0:
            right.isLeaf = True
        return left, right
    
    def featureDelete(self):
        selected = np.random.choice(feature_num, self.max_features, replace=False)
        return np.delete(np.arange(feature_num), selected)
    
    def growTree(self, tree):
        # Init tree info.
        tree.info = impurity(tree.x['label'], self.criterion)
        if tree.depth >= self.max_depth or tree.info == 0:
            tree.isLeaf = True
            return tree
        
        # Loop for preventing useless split
        for _iter in range(3):
            # Randomly select m attribute
            tmp_df = tree.x.copy()
            delete_feature = self.featureDelete()
            tmp_df = tmp_df.drop(columns=[feature_name[i] for i in delete_feature])
            
            # Select best split attribute and threshold
            info_gain, attr_index, threshold = self.splitAttr(tmp_df, tree.info)
            if info_gain == 0:
                tree.isLeaf = True
                return tree
            
            # Partition tree
            left, right = self.partiTree(tree, attr_index, threshold)
            if len(left.x) != len(tree.x) and len(right.x) != len(tree.x):
                break
        
        tree.setting(attr_index, threshold, left, right)
        
        # Move to tree's children
        if not left.isLeaf:
            self.growTree(left)
        if not right.isLeaf:
            self.growTree(right)
        return tree
    
    def Bagging(self, X):
        df = pd.DataFrame(columns=X.columns)
        for n in range(len(X)):
            index = np.random.randint(0, len(X))
            df = df.append(X.loc[index])
        return df
    
    def buildForest(self, X):
        for n in range(self.n_estimators):
            data = self.Bagging(X) if self.boostrap else X
            root = Tree(data, 1)
            self.forest.append(self.growTree(root))
                
    def model(self, test):
        vote = np.zeros((len(_class)), dtype=np.uint32)
        for n in range(self.n_estimators):
            tmp = self.forest[n]
            while True:
                if test[tmp.attr_index] <= tmp.threshold:
                    if tmp.left == None:
                        break
                    tmp = tmp.left
                else:
                    if tmp.right == None:
                        break
                    tmp = tmp.right
            pre = np.zeros((len(_class)), dtype=np.uint32)
            for c in range(len(_class)):
                pre[c] = np.sum(tmp.x['label']==_class[c])
            vote[np.argmax(pre)] += 1
        #print(vote)
        return _class[np.argmax(vote)]
        
    def predict(self, X):
        pred_y = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            pred_y[i] = self.model(X.values[i])  
        return pred_y

if __name__ == "__main__":
    data = load_breast_cancer()
    #feature_names = data['feature_names']
    #print(feature_names)
    
    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_test_df = pd.read_csv("y_test.csv")
    y_test = np.array(y_test_df['0']) # as np array
    train = pd.concat([x_train, y_train], axis=1)
    train = train.rename(columns={'0': 'label'})
    feature_name = [s for s in x_train.columns]
    
    print("------------------------------------------------------")
    # Question 1
    print("Question 1")
    data = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))
    
    print("------------------------------------------------------")
    # Question 2.1
    print("Question 2.1: Decision tree with different depth")
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth3.fit(train)
    y_pred = clf_depth3.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Criterion = Gini, Max Depth = 3, Acc:", acc)
    
    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    clf_depth10.fit(train)
    y_pred = clf_depth10.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Criterion = Gini, Max Depth = 10, Acc:", acc)
    
    print("------------------------------------------------------")
    # Question 2.2
    print("Question 2.2: Decision Tree with different criterion")
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_gini.fit(train)
    y_pred = clf_gini.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Criterion = Gini, Max Depth = 3, Acc:", acc)
    
    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    clf_entropy.fit(train)
    y_pred = clf_entropy.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Criterion = Entropy, Max Depth = 3, Acc:", acc)    
    
    print("------------------------------------------------------")
    # Question 3: Visualize
    print("Question 3: visualize as plot")
    clf_depth10.countImportance(clf_depth10.tree)
    visualize(clf_depth10.importance)
    
    print("------------------------------------------------------")
    print("Question 4.1: Random forest with different # of estimator")
    # Question 4.1: Random Forest
    clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_10tree.buildForest(train)
    y_pred = clf_10tree.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"# of estimoators = 10, Maximum # of features = 5, Acc: {acc}")
    
    clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
    clf_100tree.buildForest(train)
    y_pred = clf_100tree.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"# of estimoators = 100, Maximum # of features = 5, Acc: {acc}")    
    
    print("------------------------------------------------------")
    # Question 4.2
    print("Question 4.2: Random forest with different # of feature")
    clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_random_features.buildForest(train)
    y_pred = clf_random_features.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"# of estimoators = 10, Maximum # of features = 5, Acc: {acc}")
    
    clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])
    clf_all_features.buildForest(train)
    y_pred = clf_all_features.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"# of estimoators = 10, Maximum # of features = 30, Acc: {acc}")
    print("------------------------------------------------------")
    
    