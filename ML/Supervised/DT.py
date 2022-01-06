import numpy as np
from collections import Counter
from scipy import stats
from DTUtility import find_best_feature, partition_classes


class MyDecisionTree(object):
    def __init__(self, max_depth=2):
        """
        TODO: Initializing the tree as an empty dictionary, as preferred.
        [5 points]

        For example: self.tree = {}

        Args:

        max_depth: maximum depth of the tree including the root node.
        """
        self.tree = {'leftTree': [], 'rightTree': []}
        self.max_depth = max_depth

    def fit(self, X, y, depth):
        """
        TODO: Train the decision tree (self.tree) using the the sample X and labels y.
        [10 points]

        NOTE: You will have to make use of the utility functions to train the tree.
        One possible way of implementing the tree: Each node in self.tree could be in the form of a dictionary:
        https://docs.python.org/2/library/stdtypes.html#mapping-types-dict

        For example, a non-leaf node with two children can have a 'left' key and  a  'right' key.
        You can add more keys which might help in classification (eg. split attribute and split value)


        While fitting a tree to the data, you will need to check to see if the node is a leaf node(
        based on the stopping condition explained above) or not.
        If it is not a leaf node, find the best feature and attribute split:
        X_left, X_right, y_left, y_right, for the data to build the left and
        the right subtrees.

        Remember for building the left subtree, pass only X_left and y_left and for the right subtree,
        pass only X_right and y_right.

        Args:

        X: N*D matrix corresponding to the data points
        Y: N*1 array corresponding to the labels of the data points
        depth: depth of node of the tree

        """
        #       node = {
        #             'isLeaf': False,
        #             'split_attribute': split_attribute,
        #             'split_value': split_val,
        #             'is_categorical': is_categorical,
        #             'leftTree': leftTree,
        #             'rightTree': rightTree
        #             'depth': depth
        #              };
        #       leaf-node = {
        #             'isLeaf': True,
        #             'prediction': 1 or 0
        #         };
        if (depth == self.max_depth) or (sum(y) == len(y) or sum(y) == 0):
            self.tree = {'isLeaf': True,
                         'prediction': stats.mode(y)[0][0],
                         'depth': depth,
                         'labels': y,
                         "data": X}
            return

        else:
            best_feas, best_split_val = find_best_feature(X, y)

            # is_categoriacal
            if type(best_split_val) == str:
                self.tree['is_categorical'] = True
            else:
                self.tree['is_categorical'] = False

            self.tree['isLeaf'] = False
            self.tree['split_attribute'] = best_feas
            self.tree['split_value'] = best_split_val

            # left right
            X_left, X_right, y_left, y_right = partition_classes(X, y, best_feas, best_split_val)
            self.tree['X_left'] = X_left
            self.tree['y_left'] = y_left
            self.tree['X_right'] = X_right
            self.tree['y_right'] = y_right

            self.tree['leftTree'] = MyDecisionTree(self.max_depth)
            self.tree['rightTree'] = MyDecisionTree(self.max_depth)

            self.tree['leftTree'].fit(X_left, y_left, depth + 1)
            self.tree['rightTree'].fit(X_right, y_right, depth + 1)

    def predict(self, record):
        """
        TODO: classify a sample in test data set using self.tree and return the predicted label
        [5 points]
        Args:

        record: D*1, a single data point that should be classified

        Returns: True if the predicted class label is 1, False otherwise

        """
        # compare index attribute
        curr = self.tree

        while True:
            leafOrNot = curr.get('isLeaf')
            if leafOrNot == False:
                is_cat = curr.get('is_categorical')
                one_fea = record[curr.get('split_attribute')]
                if is_cat:
                    if one_fea == curr.get('split_value'):
                        curr = curr.get('leftTree').tree
                        continue
                    else:
                        curr = curr.get('rightTree').tree
                        continue
                else:
                    if one_fea <= curr.get('split_value'):
                        curr = curr.get('leftTree').tree
                        continue
                    else:
                        curr = curr.get('rightTree').tree
                        continue
            else:
                break

        if curr.get('prediction') == 1:
            return True
        else:
            return False

    # helper function. You don't have to modify it
    def DecisionTreeEvalution(self, X, y, verbose=False):

        # Make predictions
        # For each test sample X, use our fitting dt classifer to predict
        y_predicted = []
        for record in X:
            y_predicted.append(self.predict(record))

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        if verbose:
            print("accuracy: %.4f" % accuracy)
        return accuracy

    def DecisionTreeError(self, y):
        # helper function for calculating the error of the entire subtree if converted to a leaf with majority class label.
        # You don't have to modify it
        num_ones = np.sum(y)
        num_zeros = len(y) - num_ones
        return 1.0 - max(num_ones, num_zeros) / float(len(y))

    #  Define the post-pruning function
    def pruning(self, X, y):
        """
        TODO:
        1. Prune the full grown decision tress recursively in a bottom up manner.
        2. Classify examples in validation set.
        3. For each node:
        3.1 Sum errors over the entire subtree. You may want to use the helper function "DecisionTreeEvalution".
        3.2 Calculate the error on same example if converted to a leaf with majority class label.
        You may want to use the helper function "DecisionTreeError".
        4. If error rate in the subtree is greater than in the single leaf, replace the whole subtree by a leaf node.
        5. Return the pruned decision tree.
        """
        # Base Case node is Leaf
        if self.tree['isLeaf'] == True:
            return;
        else:
            # get parameters for split dataset
            split_attribute = self.tree['split_attribute']
            split_value = self.tree['split_value']

            # validation dataset
            X_left_val, X_right_val, y_left_val, y_right_val = partition_classes(X, y, split_attribute, split_value)
            combined_y = np.concatenate((y_left_val, y_right_val), axis=0)

            # check compare
            error = self.DecisionTreeError(combined_y)
            accuracy_left = 0
            accuracy_right = 0
            if y_left_val.any():
                accuracy_left = self.DecisionTreeEvalution(X_left_val, y_left_val)
            if y_right_val.any():
                accuracy_right = self.DecisionTreeEvalution(X_right_val, y_right_val)
            accuracy = (accuracy_left + accuracy_right)

            # compare and modify
            if (1 - accuracy) > error:
                self.tree['isLeaf'] = True
                self.tree['prediction'] = stats.mode(combined_y)[0][0]

            if self.tree['leftTree'] is not None:
                self.tree['leftTree'].pruning(X_left_val, y_left_val)
            if self.tree['rightTree'] is not None:
                self.tree['rightTree'].pruning(X_right_val, y_right_val)
        #             self.tree['leftTree'].pruning(X_left_val, y_left_val)
        #             self.tree['rightTree'].pruning(X_right_val, y_right_val)
        return self
