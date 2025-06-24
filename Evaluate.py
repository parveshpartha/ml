#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
evaluate.py

Evaluation functions for regression models.
"""

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_regression_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate regression model using RMSE and R².

    Parameters:
        model: Trained regression model
        X: Features to predict
        y: Actual target values
        dataset_name: Name for logging (Train, Validation, Test)

    Prints:
        RMSE and R² Score
    """
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)

    print(f"\n{dataset_name} Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

