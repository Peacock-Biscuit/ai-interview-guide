import numpy as np
from math import log2, sqrt

def entropy(class_y):
    """
    Input:
        - class_y: list of class labels (0's and 1's)

    TODO: Compute the entropy for a list of classes
    Example: entropy([0,0,0,1,1,1,1,1]) = 0.9544
    """
    freq_dict = {}
    for i in class_y:
        if i not in freq_dict:
            freq_dict[i] = 1 / len(class_y)
        else:
            freq_dict[i] += 1 / len(class_y)
    entropy = 0
    for i, j in freq_dict.items():
        entropy += -(j) * log2(j)
    return entropy


def information_gain(previous_y, current_y):
    """
    Inputs:
        - previous_y : the distribution of original labels (0's and 1's)
        - current_y  : the distribution of labels after splitting based on a particular
                     split attribute and split value

    TODO: Compute and return the information gain from partitioning the previous_y labels into the current_y labels.

    Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

    Example: previous_y = [0,0,0,1,1,1], current_y = [[0,0], [1,1,1,0]], info_gain = 0.4591
    """
    # IG = H – (HLx PL+ HR x PR)
    H = entropy(previous_y)
    total_entropy = 0
    for par in current_y:
        par_H = entropy(par)
        total_entropy += par_H * (len(par) / len(previous_y))
    return H - total_entropy



def partition_classes(X, y, split_attribute, split_val):
    """
    Inputs:
    - X               : (N,D) list containing all data attributes
    - y               : a list of labels
    - split_attribute : column index of the attribute to split on
    - split_val       : either a numerical or categorical value to divide the split_attribute

    TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.

    Example:

    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]

    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.

    Consider the case where we call the function with split_attribute = 0 (the index of attribute) and split_val = 3 (the value of attribute).
    Then we divide X into two lists - X_left, where column 0 is <= 3 and X_right, where column 0 is > 3.

    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]

    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.

    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]

    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]


    Return in this order: X_left, X_right, y_left, y_right
    """
    X = np.array(X, dtype=object)
    y = np.array(y)
    X_left, X_right, y_left, y_right = np.array([]), np.array([]), np.array([]), np.array([])
    if type(split_val) is str:
        X_left = X[X[:, split_attribute] == split_val]
        y_left = y[X[:, split_attribute] == split_val]
        X_right = X[X[:, split_attribute] != split_val]
        y_right = y[X[:, split_attribute] != split_val]
    else:
        X_left = X[X[:, split_attribute] <= split_val]
        y_left = y[X[:, split_attribute] <= split_val]
        X_right = X[X[:, split_attribute] > split_val]
        y_right = y[X[:, split_attribute] > split_val]
    return np.array(X_left), np.array(X_right), np.array(y_left), np.array(y_right)

def find_best_split(X, y, split_attribute):
    """
    Inputs:
        - X               : (N,D) list containing all data attributes
        - y               : a list array of labels
        - split_attribute : Column of X on which to split

    TODO: Compute and return the optimal split value for a given attribute, along with the corresponding information gain

    Note: You will need the functions information_gain and partition_classes to write this function

    Example:

        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]

        split_attribute = 0

        Starting entropy: 0.971

        Calculate information gain at splits:
           split_val = 1  -->  info_gain = 0.17
           split_val = 2  -->  info_gain = 0.01997
           split_val = 3  -->  info_gain = 0.01997
           split_val = 4  -->  info_gain = 0.32
           split_val = 5  -->  info_gain = 0

       best_split_val = 4; info_gain = .32;
    """
    X = np.array(X, dtype=object)
    y = np.array(y)
    unique = set()
    best_val = None
    max_gain = -1 # 0 <= info gain, so it is safe to set it as negative

    for val in np.unique(X[:, split_attribute]):
        if val not in unique:
            _, _, y_left, y_right = partition_classes(X, y, split_attribute, val)
            ig = information_gain(y, [y_left, y_right])
            if ig > max_gain:
                max_gain = ig
                best_val = val
            unique.add(val)
    return best_val, max_gain


def find_best_feature(X, y):
    """
    Inputs:
        - X: (N,D) list containing all data attributes
        - y : a list of labels

    TODO: Compute and return the optimal attribute to split on and optimal splitting value

    Note: In find_best_feature, choose the first feature if two features tie.

    Example:

        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]

        split_attribute = 0

        Starting entropy: 0.971

        Calculate information gain at splits:
           feature 0:  -->  info_gain = 0.32
           feature 1:  -->  info_gain = 0.17
           feature 2:  -->  info_gain = 0.4199

       best_split_feature: 2 best_split_val: 22
    """
    X = np.array(X, dtype=object)
    y = np.array(y)
    best_feas = None
    best_split_val = None
    ig_max = -1

    for i in range(len(X[0])):
        split_val, ig_fea = find_best_split(X, y, i)
        if ig_fea > ig_max:
            ig_max = ig_fea
            best_feas = i
            best_split_val = split_val
    return best_feas, best_split_val
