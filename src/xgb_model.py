# src/xgb_model.py

import xgboost as xgb
import numpy as np


class XGBModel:
    def __init__(self, **kwargs):
        """
        Initialize XGBoost classifier with keyword arguments,
        e.g. n_estimators=100, max_depth=6, etc.

        Note: Removed 'use_label_encoder' because it's deprecated.
        """
        self.model = xgb.XGBClassifier(eval_metric='logloss', **kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost with optional validation data for progress reporting.
        """
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=True  # shows progress
            )
        else:
            # If no validation set, just train
            self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """
        Returns predicted probabilities (Nx2 for binary classification).
        """
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
