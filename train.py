import xgboost as xgb

def train_xgboost_regressor(X_train, y_train, X_val, y_val, early_stopping_rounds=50):
    """
    Train XGBoost Regressor with validation monitoring.

    Returns:
        model: Trained XGBoost model
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )

    return model
