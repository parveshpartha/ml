def train_xgboost_regressor(X_train, y_train):
    """
    Train an XGBoost Regressor.

    Parameters:
        X_train (array-like): Training features
        y_train (array-like): Target values

    Returns:
        model (xgb.XGBRegressor): Trained model
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
    
    return model  # << YOU MUST HAVE THIS LINE
