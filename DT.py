import math
import numpy as np
from collections import Counter
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, val=None) -> None:
        # feature this was split on
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.return_val = val
    
    def is_leaf_node(self):
        return self.return_val is not None

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None) -> None:
        self.min_sample_split = min_sample_split # dont split node if fewer then this num examples
        self.max_depth = max_depth 
        self.n_features = n_features # only consider the first (_) number of features
        self.root = None # root node of tree
    
    def predict(self, X):
        # get prediction for every X being passed in
        return np.array([self._traverse(self.root, x) for x in X])

    def _traverse(self, node, x):
        if node.is_leaf_node():
            return node.return_val
        
        if x[node.feature] < node.threshold:
            return self._traverse(node.left, x)
        else:
            return self._traverse(node.right, x)
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y, 0)
    
    def _grow_tree(self, X, y, depth):
        
        # check stopping criterion
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split:
            leaf_value = self._most_common_label(y)
            return Node(val=leaf_value) # setting leaf node to be majority vote
        
        # only want to be splitting on a subset of the features 
        features_to_split = np.random.choice(n_feats, self.n_features, replace=False)
        
        # find the best split by maximizing information gain
        best_threshold, best_feature = self._best_split(X,y,features_to_split)

        # create children nodes
        left_idxs, right_idxs = self._split(best_threshold, X[:,best_feature])

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)


    def _best_split(self, X, y, features_to_split):
        # loop over all splits and thresholds 
        best_gain = -1
        split_feature, split_threshold = None, None

        for feature in features_to_split:
            for threshold in np.unique(X[:,feature]): # looping over all values that this feature can take in dataset

                # split the data based on this feature and threshold and compute the information gain
                information_gain = self._information_gain(threshold, X[:, feature], y)
                if information_gain > best_gain:
                    best_gain = information_gain
                    split_feature = feature
                    split_threshold = threshold
        
        return split_threshold, split_feature

    def _information_gain(self, threshold, X_col, y):

        # compute parent entropy
        parent_entropy = self._entropy(y)
        
        # make 2 new datasets based on feature, threshold
        left_idxs, right_idxs = self._split(threshold, X_col)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # get entropy of the children nodes after split
        left_entropy, right_entropy = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # compute weighted entropy of children
        weighted_entropy = (len(left_idxs)/len(y))*left_entropy + (len(right_idxs)/len(y))*right_entropy
        
        return parent_entropy - weighted_entropy
        
        

    def _split(self, threshold, X_col):
        # return a 2 list of indexs representing the elements in X that are above/below threshold
        left_idxs = np.argwhere(X_col <= threshold).flatten() # left child smaller by convention
        right_idxs = np.argwhere(X_col > threshold).flatten()
        return left_idxs, right_idxs

    
    def _entropy(self, y):
        # entropy is defined with respect to the labels only; has nothing to do with the features
        histogram = np.bincount(y) # computing pi for each label
        prob_distribution = histogram / len(y)
        entropy = 0
        for p in prob_distribution:
            if p == 0: continue
            entropy += p * math.log2(p)
        return -entropy
        

    def _most_common_label(self, y):
        # assuming we are doing classification and not regression; taking majority vote of label
        value = max([(freq,val) for val, freq in Counter(y).items()])[1]
        return value