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




