# Import Libraries
import pandas as pd
import numpy as np
# import xgboost as xgb
# import seaborn as sns
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OrdinalEncoder
from .utils_baseline import get_individual_thresholds, \
                            get_prediction_sets_individually_thresholded, \
                            compile_predictions, \
                            get_coverage_and_set_size, \
                            get_weighted_coverage, \
                            get_weighted_set_size
                            # get_prediction_set_labels, \
# from .utils_preprocess import *
from .utils_perturbation import provide_categorical_perturbation_to_data_sets, \
                                provide_numerical_perturbation_to_datasets, \
                                get_perturbed_datasets_summary
from .utils_visualization import get_coverage_based_feature_importance, \
                                get_set_size_based_feature_importance, \
                                get_feature_importance_per_class

def calibrate_and_return_performance( classifier, n_classes, X_test, X_Cal, y_test, y_cal, alpha, class_labels):
    """
    Calibrate the model and return performance metrics.

    Args:
        classifier (object): Classifier object used for prediction.
        n_classes (int): Number of classes in the dataset.
        X_test (pandas.DataFrame): Test set.
        X_Cal (pandas.DataFrame): Calibration set.
        y_test (pandas.Series): True class labels for the test set.
        y_cal (pandas.Series): True class labels for the calibration set.
        alpha (float): Significance level.
        class_labels (list): List of class labels.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame, list, float, float]: Tuple containing results sets, results,
        thresholds, weighted coverage, and weighted set size.
    """
    thresholds = get_individual_thresholds(alpha, n_classes, X_Cal, y_cal, classifier)
    #print('calibrate and return' + str(n_classes))
    prediction_sets = get_prediction_sets_individually_thresholded(classifier, X_test, thresholds, n_classes)
    # prediction_set_labels = get_prediction_set_labels(prediction_sets, class_labels)
    results_sets = compile_predictions(X_test, y_test, classifier, prediction_sets, class_labels)
    results = get_coverage_and_set_size(class_labels, y_test, prediction_sets, n_classes)
    weighted_coverage = get_weighted_coverage(results['Coverage'], results['Total Data Instances Per Class'])
    weighted_set_size = get_weighted_set_size(results['Average Set Size'], results['Total Data Instances Per Class'])

    return results_sets, results, thresholds, weighted_coverage, weighted_set_size

def calculate_relative_changes(perturbed_df, org_weighted_coverage, org_weighted_set_size):
    """
    Calculate relative changes in coverage and set size.

    Args:
        perturbed_df (pandas.DataFrame): DataFrame containing perturbed datasets.
        org_weighted_coverage (float): Original (baseline) weighted coverage.
        org_weighted_set_size (float): Original (baseline) weighted set size.

    Returns:
        pandas.DataFrame: DataFrame containing relative changes in coverage and set size.
    """
    # Now we will calculate the Relative Change in Weighted Coverage in Each Column Compared to the Original(baseline) Weighted Coverage
    # Consider later negative and positive changes maybe we can put directions like in LIME as well

    perturbed_df['Relative Change in Coverage'] = np.abs((perturbed_df['Weighted Coverage'] - org_weighted_coverage)/org_weighted_coverage * 100)
    perturbed_df['Relative Change in Set Size'] = np.abs((perturbed_df['Weighted Set Size'] - org_weighted_set_size)/org_weighted_set_size * 100)

    return perturbed_df

# A function to run the Conformal Scores and Resulting Coverage and Average Size Sets for each Dataset to compare it with original
def produce_performance_metrics_over_multiple_datasets(perturbed_df, classifier, X_test, y_test, y_cal, alpha, class_labels):
    """
    Produce performance metrics over multiple datasets.

    Args:
        perturbed_df (pandas.DataFrame): DataFrame containing perturbed datasets.
        classifier (object): Classifier object used for prediction.
        X_test (pandas.DataFrame): Test set.
        y_test (pandas.Series): True class labels for the test set.
        y_cal (pandas.Series): True class labels for the calibration set.
        alpha (float): Significance level.
        class_labels (list): List of class labels.

    Returns:
        Tuple[pandas.DataFrame, list]: Tuple containing perturbed DataFrame and list of prediction sets.
    """
    ls_pred_sets = []
    ls_results = []
    ls_thresholds = []
    ls_coverage = []
    ls_avg_set_size = []
    ls_weighted_coverage = []
    ls_weighted_set_size = []
    n_classes = len(class_labels)
    # Iterate through each row of the compiled DataFrame
    for _, row in perturbed_df.iterrows():
        # Retrieve information from the current row
        perturbed_data = row['Perturbed_Data']
        # perturbed_column = row['Perturbed_Column']
        # perturbation_type = row['Perturbation_Type']
        # perturbation_severity = row['Perturbation_Severity']

        X_Cal = perturbed_data.copy()

        # Extract text features
        cats = X_Cal.select_dtypes(exclude=np.number).columns.tolist()

        # Convert to pd.Categorical
        for col in cats:
            X_Cal[col] = X_Cal[col].astype('category')

        # Now you can use 'perturbed_data', 'perturbed_column', etc. in your further analysis or processing
        # For example, print the perturbed column name and first few rows of the perturbed data
        #print(f"Perturbed Column: {perturbed_column}")
        results_sets, results, threshold, weighted_coverage, weighted_set_size = calibrate_and_return_performance( classifier, n_classes,  X_test, X_Cal, y_test, y_cal, alpha, class_labels)

        ls_pred_sets.append(results_sets)
        ls_results.append(results)
        ls_thresholds.append(threshold)
        ls_coverage.append(results["Coverage"])
        ls_avg_set_size.append(results["Average Set Size"])
        ls_weighted_coverage.append(weighted_coverage)
        ls_weighted_set_size.append(weighted_set_size)


        # Call another function or perform additional analysis with 'perturbed_data'
        # e.g., process_perturbed_data(perturbed_data)
    perturbed_df['Coverage'] = ls_coverage
    perturbed_df['Average Set Size'] = ls_avg_set_size
    perturbed_df['Threshold'] = ls_thresholds
    perturbed_df['Weighted Coverage'] = ls_weighted_coverage
    perturbed_df['Weighted Set Size'] = ls_weighted_set_size
    perturbed_df['Prediction Sets'] = ls_pred_sets

    return perturbed_df, ls_pred_sets


def compile_relative_changes_per_class(global_summary_table, results_base_line, labels):
    """
    Compile relative changes per class.

    Args:
        global_summary_table (pandas.DataFrame): DataFrame containing global summary table.
        results_base_line (pandas.DataFrame): DataFrame containing baseline results.
        labels (list): List of class labels.

    Returns:
        pandas.DataFrame: DataFrame containing relative changes per class.
    """
    list_of_relative_changes_per_class_coverage = []
    list_of_relative_changes_per_class_set_size = []

    for i in range(len(global_summary_table)):
        coverage_sample = np.abs((global_summary_table.iloc[i]['Coverage'].values - results_base_line['Coverage'].values)/ results_base_line['Coverage'].values  * 100)
        set_size_sample = np.abs((global_summary_table.iloc[i]['Average Set Size'].values - results_base_line['Average Set Size'].values)/results_base_line['Average Set Size'].values * 100)

        # Create a DataFrame from the lists
        df_covg = pd.DataFrame({'Class': labels, 'Relative Change in Coverage Per Class': coverage_sample})
        df_set_size = pd.DataFrame({'Class': labels, 'Relative Change in Set Size Per Class': set_size_sample})
        
        list_of_relative_changes_per_class_coverage.append(df_covg)
        list_of_relative_changes_per_class_set_size.append(df_set_size)

    global_summary_table['Relative Change in Coverage Per Class'] = list_of_relative_changes_per_class_coverage
    global_summary_table['Relative Change in Set Size Per Class'] = list_of_relative_changes_per_class_set_size

    return global_summary_table
    
##### if you want to see mapping as a table
def compile_feature_perturbed_and_classes_mapping(global_summary_table, class_labels):
    """
    Compile feature perturbed and classes mapping.

    Args:
        global_summary_table (pandas.DataFrame): DataFrame containing global summary table.
        class_labels (list): List of class labels.

    Returns:
        None
    """
    exp_table = global_summary_table[['Perturbed_Column', 'Relative Change in Coverage Per Class', 'Relative Change in Set Size Per Class']]

    for feature_perturbed in list(np.unique(exp_table['Perturbed_Column'])):
        sub_sum = exp_table[ exp_table['Perturbed_Column'] == feature_perturbed ]['Relative Change in Coverage Per Class']
        for class_name in class_labels:
            for row_no in range(len(sub_sum)):
                ls_spec = []
                spec_table = sub_sum.iloc[row_no]
                ls_spec.append(spec_table[spec_table['Class'] == class_name]['Relative Change in Coverage Per Class'])


def get_perturbed_features_and_class_pairs(global_summary_table, class_labels):
    """
    Get perturbed features and class pairs.

    Args:
        global_summary_table (pandas.DataFrame): DataFrame containing global summary table.
        class_labels (list): List of class labels.

    Returns:
        pandas.DataFrame: DataFrame containing perturbed features and class pairs.
    """
    exp_table = global_summary_table[['Perturbed_Column', 'Relative Change in Coverage Per Class', 'Relative Change in Set Size Per Class']]

    # Create an empty list to store dictionaries representing each row
    rows = []

    for feature_perturbed in np.unique(exp_table['Perturbed_Column']):
        sub_sum = exp_table[exp_table['Perturbed_Column'] == feature_perturbed]['Relative Change in Coverage Per Class']

        for class_name in class_labels:
            ls_spec = []

            for row_no in range(len(sub_sum)):
                spec_table = sub_sum.iloc[row_no]
                ls_spec.append(spec_table[spec_table['Class'] == class_name]['Relative Change in Coverage Per Class'].values[0])

            # Calculate mean for each class
            mean_value = np.mean(ls_spec)

            # Create a dictionary representing the row
            row_dict = {
                'Feature Perturbed': feature_perturbed,
                'Class': class_name,
                'Mean Relative Change': mean_value
            }

            # Append the dictionary to the list
            rows.append(row_dict)

    # Create a DataFrame from the list of dictionaries
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(by='Mean Relative Change', ascending=False)

    return result_df

def get_global_summary_table_per_class(perturbed_df, results_base_line, class_labels):
    """
    Get global summary table per class.

    Args:
        perturbed_df (pandas.DataFrame): DataFrame containing perturbed datasets.
        results_base_line (pandas.DataFrame): DataFrame containing baseline results.
        class_labels (list): List of class labels.

    Returns:
        pandas.DataFrame: DataFrame containing global summary table per class.
    """
    global_summary_table = perturbed_df[['Perturbed_Column','Coverage', 'Average Set Size' ]].copy()
    global_summary_table = compile_relative_changes_per_class(global_summary_table, results_base_line, class_labels)
    #result_df = get_perturbed_features_and_class_pairs(global_summary_table, class_labels)

    return global_summary_table 

def plot_conformal_explainer(classifier, X_Cal, y_cal, X_test, y_test, alpha, class_labels, type_of_metric, noise_type):
    """
    Plot conformal explainer.

    Args:
        classifier (object): Classifier object used for prediction.
        X_Cal (pandas.DataFrame): Calibration set.
        y_cal (pandas.Series): True class labels for the calibration set.
        X_test (pandas.DataFrame): Test set.
        y_test (pandas.Series): True class labels for the test set.
        alpha (float): Significance level.
        class_labels (list): List of class labels.
        type_of_metric (str): Type of metric to visualize.
        noise_type (str): Type of noise applied.

    Returns:
        Depends on type_of_metric parameter.
    """
    # produce baseline metrics
    n_classes = len(np.unique(y_test))
    #print('plot cnf explainer' + str(n_classes))
    _, results, _, weighted_coverage, weighted_set_size = calibrate_and_return_performance( classifier, n_classes,  X_test, X_Cal, y_test, y_cal, alpha, class_labels)
    # provide counterfactual perturbations
    list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, list_of_df = provide_categorical_perturbation_to_data_sets(X_Cal)
    list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, list_of_df = provide_numerical_perturbation_to_datasets(X_Cal, list_of_df, list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, noise_type)
    # summarize perturbed datasets
    perturbed_df = get_perturbed_datasets_summary(list_of_perturb_column, list_of_perturb_type, list_of_perturb_severity, list_of_df)
    perturbed_df, _ = produce_performance_metrics_over_multiple_datasets(perturbed_df, classifier, X_test, y_test, y_cal, alpha, class_labels)
    # calculate relative changes
    perturbed_df = calculate_relative_changes(perturbed_df, weighted_coverage, weighted_set_size)

    # visualize according to the type
    if type_of_metric == "coverage":
        #return get_coverage_based_feature_importance(perturbed_df).sort_values(by='Relative Change in Coverage', ascending=False)
        # Get coverage-based feature importance
        #print(perturbed_df)
        importance_df = get_coverage_based_feature_importance(perturbed_df)
        #print(importance_df)

        # Filter out lines with 0.0 relative change
        importance_df_filtered = importance_df[importance_df['Relative Change in Coverage'] != 0.0]

        # Sort the DataFrame by 'Relative Change in Coverage' in descending order
        importance_df_sorted = importance_df_filtered.sort_values(by='Relative Change in Coverage', ascending=False)
        return importance_df_sorted


    if type_of_metric == "set_size":
        return get_set_size_based_feature_importance(perturbed_df).sort_values(by='Relative Change in Set Size', ascending=False)

    if type_of_metric == "pred_set":
        global_summary_table = get_global_summary_table_per_class(perturbed_df, results, class_labels)
        result_df = get_perturbed_features_and_class_pairs(global_summary_table, class_labels)
        # result_df = result_df[result_df['Mean Relative Change'] != 0.0]
        get_feature_importance_per_class(result_df)

        # Compile resulting data frame and group the perturbed features
        result_df = result_df.sort_values(by='Mean Relative Change', ascending=False).groupby('Class')
        # Iterate over groups and print the group keys and the corresponding DataFrame
        for group_key, group_df in result_df:
            print("Group:", group_key)
            print(group_df)
            print()
