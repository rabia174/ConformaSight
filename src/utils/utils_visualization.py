# Import Libraries
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# import pandas as pd
# import numpy as np
# import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OrdinalEncoder
# from .utils_baseline import *
# from .utils_preprocess import *
# from .utils_explain import *
# from .utils_perturbation import *
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config.ini file
# Please set the path properly before usage
config.read('../configurations/config.ini')

WIDTH = config.getint('figure_size_variables', 'WIDTH')
HEIGHT = config.getint('figure_size_variables', 'HEIGHT')

def get_coverage_based_feature_importance(perturbed_df):
    """
    Get coverage-based feature importance and visualize it.

    Args:
        perturbed_df (pandas.DataFrame): DataFrame containing perturbed datasets and their characteristics.

    Returns:
        pandas.DataFrame: Summary of feature importance based on coverage.
    """
    # Now we can get the mean by column or in the future get the max value of relative change to see obvious changes
    summary = perturbed_df[['Perturbed_Column', 'Relative Change in Coverage']]
    summary = summary.groupby("Perturbed_Column").mean()
    summary = summary[summary['Relative Change in Coverage'] != 0.0]
    summary = summary.sort_values(by='Relative Change in Coverage', ascending=True)

    summary.plot(kind='barh', figsize=(WIDTH, HEIGHT), legend=True)
    plt.title('Conformal Feature Importance (Based on Coverage)')
    plt.xlabel('Importance Weights')
    plt.ylabel('Features')
    # Get the current Axes instance
    ax = plt.gca()
    
    # Turn on the grid
    ax.grid(True)
    
    # Show the plot
    plt.show()

    return summary

def get_set_size_based_feature_importance(perturbed_df):
    """
    Get set size-based feature importance and visualize it.

    Args:
        perturbed_df (pandas.DataFrame): DataFrame containing perturbed datasets and their characteristics.

    Returns:
        pandas.DataFrame: Summary of feature importance based on set size.
    """
    summary = perturbed_df[['Perturbed_Column', 'Relative Change in Set Size']]
    summary = summary.groupby("Perturbed_Column").mean()
    summary = summary[summary['Relative Change in Set Size'] != 0.0]
    summary = summary.sort_values(by='Relative Change in Set Size', ascending=True)

    summary.plot(kind='barh', figsize=(WIDTH, HEIGHT), legend=True)
    plt.title('Conformal Feature Importance (Based on Set Size)')
    plt.xlabel('Importance Weights')
    plt.ylabel('Features')
    ax = plt.gca()
    
    # Turn on the grid
    ax.grid(True)
    
    # Show the plot
    plt.show()

    return summary

def get_feature_importance_per_class(result_df):
    """
    Get feature importance per class and visualize it.

    Args:
        result_df (pandas.DataFrame): DataFrame containing perturbed features and their relative changes per class.

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=result_df, kind="bar", orient='h',
        y="Feature Perturbed", x="Mean Relative Change", hue="Class",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    
    g.despine(left=True)
    g.set_axis_labels("Feature Importance Weights", "Perturbed Features")
    g.legend.set_title("Classes")
    # Activate the grid
    plt.grid(True)
    plt.figure(figsize=(WIDTH, HEIGHT))
    
    # Show the plot
    plt.show()
    