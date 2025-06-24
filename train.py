import xgboost as xgb

def train_xgboost_regressor(X_train, y_train, X_val, y_val, early_stopping_rounds=10):
    """
    Train XGBoost Regressor optimized for small datasets with early stopping.

    Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        early_stopping_rounds (int): Stop if no improvement after N rounds

    Returns:
        Trained XGBoost model
    """
    model = xgb.XGBRegressor(
        n_estimators=300,           # Reduced from 1000 for speed on small data
        learning_rate=0.1,
        max_depth=4,                # Slightly shallower trees to prevent overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
        callbacks=[xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]
    )

    return model
