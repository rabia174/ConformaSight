# Import Libraries

import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from keras.models import Sequential


def get_qhat(y_cal_prob_class, alpha, mask):
    """
    Calculate s_scores and adjusted quantile (qhat) based on given parameters.
    
    Args:
        y_cal_prob_class (numpy.ndarray): Array containing predicted probabilities for the calibration set.
        alpha (float): Significance level.
        mask (numpy.ndarray): Mask indicating instances of a specific class.
    
    Returns:
        Tuple[numpy.ndarray, float]: Tuple containing s_scores and adjusted quantile (qhat).
    """
    s_scores = 1- y_cal_prob_class
    q = ( 1 - alpha )*100
    # 2: get adjusted quantile
    #q_level = np.ceil((n+1)*(1-alpha))/n
    #qhat = np.quantile(cal_scores, q_level, method='higher')
    
    class_size = mask.sum()
    #print(class_size)
    correction = ( class_size + 1 ) / class_size # correct it with ceiling function later

    q *= correction
    return s_scores, q

'''

def get_qhat(y_cal_prob_class, alpha, mask):
    s_scores = 1 - y_cal_prob_class

    # Calculate the quantile level
    n = len(y_cal_prob_class)
    q_level = (n + 1 - np.ceil((n + 1) * alpha)) / n  # Adjusted quantile level to ensure it's in [0, 1]
    print(q_level)
    # Calculate qhat using np.quantile
    qhat = np.quantile(s_scores, q=q_level, interpolation='higher')
    
    return s_scores, qhat
'''


def get_y_cal_prob_class(y_cal, y_cal_prob, class_label):
    """
    Extract predicted probabilities for a specific class from the calibration set.

    Args:
        y_cal (pandas.Series): True class labels for the calibration set.
        y_cal_prob (numpy.ndarray): Predicted probabilities for the calibration set.
        class_label (int): Label of the class to extract probabilities for.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the mask and predicted probabilities for the specified class.
    """
    mask = y_cal == class_label # producing true/false masks for each class
    #mask = mask.ravel()
    mask = mask.to_numpy()
    y_cal_prob_class = y_cal_prob[mask][:, class_label] # Extracting the softmax scores of the True class
    
    return mask, y_cal_prob_class

def get_individual_thresholds(alpha, n_classes, X_Cal, y_cal, classifier):
    """
    Calculate individual thresholds for each class based on a given significance level.

    Args:
        alpha (float): Significance level.
        n_classes (int): Number of classes in the dataset.
        X_Cal (pandas.DataFrame): Calibration set.
        y_cal (pandas.Series): True class labels for the calibration set.
        classifier (object): Classifier object used for prediction.

    Returns:
        List[float]: List of individual thresholds for each class.
    """
    thresholds = []
    # Get predicted probabilities for calibration set
    if isinstance(classifier, Sequential):
       y_cal_prob = classifier.predict(X_Cal)
    else:
       y_cal_prob = classifier.predict_proba(X_Cal)
    
    # Get 95th percentile score for each class's s-scores separetly
    for class_label in range(n_classes):
        mask, y_cal_prob_class = get_y_cal_prob_class(y_cal, y_cal_prob, class_label)
        s_scores, q = get_qhat(y_cal_prob_class, alpha, mask)                                                                 
        threshold = np.percentile(s_scores, q)
        thresholds.append(threshold)
    return thresholds

# Get s_i scores for test set
def get_prediction_sets_individually_thresholded(classifier, X_test, thresholds, n_classes):
    """
    Get prediction sets for each class based on individual thresholds.

    Args:
        classifier (object): Classifier object used for prediction.
        X_test (pandas.DataFrame): Test set.
        thresholds (list): List of individual thresholds for each class.
        n_classes (int): Number of classes in the dataset.

    Returns:
        numpy.ndarray: Array containing prediction sets for each class.
    """
    # if the estimator is a sequential object
    if isinstance(classifier, Sequential):
       predicted_proba = classifier.predict(X_test)
    else:
       predicted_proba = classifier.predict_proba(X_test)
        
    si_scores = 1 - predicted_proba
    # For each class, check whether each instance is below the threshold
    prediction_sets = []
    for i in range(n_classes):
        prediction_sets.append( si_scores[:,i] <= thresholds[i] )
    prediction_sets = np.array(prediction_sets).T
    
    return prediction_sets

# Collate predictions
def compile_predictions(X_test, y_test, classifier, prediction_sets, class_labels):
    """
    Compile predictions and observed class labels into a DataFrame.

    Args:
        X_test (pandas.DataFrame): Test set.
        y_test (pandas.Series): True class labels for the test set.
        classifier (object): Classifier object used for prediction.
        prediction_sets (numpy.ndarray): Array containing prediction sets for each class.
        class_labels (list): List of class labels.

    Returns:
        pandas.DataFrame: DataFrame containing observed class labels, predicted labels, and prediction sets.
    """
    if isinstance(classifier, Sequential):
        y_pred = np.argmax(classifier.predict(X_test), axis=1)
    else:
        y_pred = classifier.predict(X_test)
    
    #y_test = y_test.ravel() # formatting y_cal
    y_test = y_test.to_numpy()
    y_test = y_test.astype(int)

    
    results_sets = pd.DataFrame()
    results_sets['observed'] = [class_labels[i] for i in y_test]
    results_sets['labels'] = get_prediction_set_labels(prediction_sets, class_labels)
    results_sets['classifications'] = [class_labels[i] for i in y_pred]

    return results_sets

# Check Coverage ans Set Size Across Classes
def get_coverage_and_set_size(class_labels, y_test, prediction_sets, n_classes):
    """
    Calculate coverage and average set size across classes.

    Args:
        class_labels (list): List of class labels.
        y_test (pandas.Series): True class labels for the test set.
        prediction_sets (numpy.ndarray): Array containing prediction sets for each class.
        n_classes (int): Number of classes in the dataset.

    Returns:
        pandas.DataFrame: DataFrame containing coverage and average set size for each class.
    """
    results = pd.DataFrame(index=class_labels)
    #print('in coverage func '+str(len(np.unique(y_test))))
    #print(np.unique(y_test))
    results['Total Data Instances Per Class'] = get_class_counts(y_test)
    results['Coverage'] = get_coverage_by_class(prediction_sets, y_test, n_classes)
    results['Average Set Size'] = get_average_set_size(prediction_sets, y_test, n_classes)
    
    return results

# A function to return prediction set labels
def get_prediction_set_labels(prediction_sets, class_labels):
    """
    Get labels for each prediction set.

    Args:
        prediction_sets (numpy.ndarray): Array containing prediction sets for each class.
        class_labels (list): List of class labels.

    Returns:
        list: List of labels for each prediction set.
    """
    # Get set of class labels for each instance in prediction sets
    prediction_set_labels = [ [class_labels[i] for i, x in enumerate(prediction_set) if x] for prediction_set in prediction_sets]

    return prediction_set_labels

# A function returns how many instances exists for each class
def get_class_counts(y_test):
    """
    Get the number of instances for each class.

    Args:
        y_test (pandas.Series): True class labels for the test set.

    Returns:
        list: List containing the number of instances for each class.
    """
    # Defining an empty list
    class_counts = [] 
    n_classes = len(np.unique(y_test))
    #print('first n_classes: ', str(n_classes))
                    
    for i in range(n_classes):
        class_counts.append(np.sum(y_test == i))
    #print('class counts')
    #print(class_counts)
    return class_counts



'''

def get_coverage_by_class(prediction_sets, y_test, n_classes):
    coverage = []
    for i in range(n_classes):
        class_coverage_sum = 0
        class_instance_count = 0
        for j, y in enumerate(y_test):
            if y == i:
                class_coverage_sum += prediction_sets[j, i]
                class_instance_count += 1
        class_coverage = class_coverage_sum / class_instance_count if class_instance_count > 0 else 0
        coverage.append(class_coverage)
    return coverage

def get_average_set_size(prediction_sets, y_test, n_classes):
    average_set_size = []
    
    for i in range(n_classes):
        class_indices = [j for j in range(len(y_test)) if y_test[j] == i]  # Get indices where y_test equals i
        class_instances = prediction_sets[class_indices, :]
        class_set_size = np.sum(class_instances, axis=1)
        class_average_set_size = np.mean(class_set_size) if len(class_set_size) > 0 else 0
        average_set_size.append(class_average_set_size)
        
    return average_set_size

'''
# A function that returns the coverage level for each class
def get_coverage_by_class(prediction_sets, y_test, n_classes):
    """
    Calculate the coverage level for each class.

    Args:
        prediction_sets (numpy.ndarray): Array containing prediction sets for each class.
        y_test (pandas.Series): True class labels for the test set.
        n_classes (int): Number of classes in the dataset.

    Returns:
        list: List containing the coverage level for each class.
    """
    # Defining an empty list
    coverage = []
    
    for i in range(n_classes):
        coverage.append(np.mean(prediction_sets[y_test == i, i]))
    return coverage
    
# A function that returns average set size for each class
def get_average_set_size(prediction_sets, y_test, n_classes):
    """
    Calculate the average set size for each class.

    Args:
        prediction_sets (numpy.ndarray): Array containing prediction sets for each class.
        y_test (pandas.Series): True class labels for the test set.
        n_classes (int): Number of classes in the dataset.

    Returns:
        list: List containing the average set size for each class.
    """
    # Defining an empty set
    average_set_size = []
    
    for i in range(n_classes):
        average_set_size.append(np.mean(np.sum(prediction_sets[y_test == i], axis=1)))
        
    return average_set_size

# Get weighted coverage (weighted by class size)
# Class counts holds the total elements in one class
def get_weighted_coverage(coverage, class_counts):
    """
    Calculate weighted coverage based on class size.

    Args:
        coverage (list): List containing the coverage level for each class.
        class_counts (list): List containing the number of instances for each class.

    Returns:
        float: Weighted coverage.
    """
    # retrieving sum of all class counts first
    total_counts = np.sum(class_counts)
    weighted_coverage = np.sum((coverage*class_counts) / total_counts)
    #weighted_coverage = round(weighted_coverage, 3)

    return weighted_coverage

# Get weighted set_size (weighted by class size)
def get_weighted_set_size(set_size, class_counts):
        """
    Calculate weighted set size based on class size.

    Args:
        set_size (list): List containing the average set size for each class.
        class_counts (list): List containing the number of instances for each class.

    Returns:
        float: Weighted set size.
    """
    total_counts = np.sum(class_counts)
    weighted_set_size = np.sum((set_size*class_counts) / total_counts)
    #weighted_set_size = round(weighted_set_size, 3)

    return weighted_set_size

