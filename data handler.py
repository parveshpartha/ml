#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
data_loader.py

Reusable functions to load pre-split datasets for ML pipelines.
"""

import pandas as pd

def load_dataset(file_path):
    """
    Load a dataset from CSV file.

    Parameters:
        file_path (str): Path to CSV file

    Returns:
        DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)


def split_features_target(df, target_column="target"):
    """
    Separate features and target from dataset.

    Parameters:
        df (DataFrame): Full dataset
        target_column (str): Name of target column

    Returns:
        X (DataFrame): Features
        y (Series): Target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

