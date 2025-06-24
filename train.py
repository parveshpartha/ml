#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
XGBoost Regressor Training Script

This script trains an XGBoost Regressor using provided feature and target data.
"""

import xgboost as xgb


def train_xgboost_regressor(X_train, y_train):
    """
    Train an XGBoost Regressor.

    Parameters:
        X_train (array-like): Training features
        y_train (array-like): Target values
        

    Returns:
        model (xgb.XGBRegressor): Trained XGBoost Regressor
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

