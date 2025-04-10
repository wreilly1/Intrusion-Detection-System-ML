# src/stack_ensemble.py

import numpy as np
from sklearn.linear_model import LogisticRegression


class StackedEnsembleModel:
    def __init__(self, xgb_params=None, mlp_params=None):
        from xgb_model import XGBModel
        from torch_model import MLPModel

        if xgb_params is None:
            xgb_params = {}
        if mlp_params is None:
            mlp_params = {}

        self.xgb_model = XGBModel(**xgb_params)
        self.mlp_model = MLPModel(**mlp_params)

        self.meta_clf = LogisticRegression()

    def fit(self, X_train, y_train):
        """
        Train the two base models on the entire training set,
        then train the meta-classifier on their predicted probabilities.
        """
        # 1. Fit XGBoost
        self.xgb_model.fit(X_train, y_train)

        # 2. Fit PyTorch MLP (shows TQDM bars)
        self.mlp_model.fit(X_train, y_train)

        # 3. Collect predictions from both
        xgb_probas = self.xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        mlp_probas = self.mlp_model.predict_proba(X_train)[:, 1].reshape(-1, 1)

        meta_features = np.hstack([xgb_probas, mlp_probas])

        # 4. Train meta-classifier (LogReg)
        self.meta_clf.fit(meta_features, y_train)

    def predict_proba(self, X):
        xgb_probas = self.xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
        mlp_probas = self.mlp_model.predict_proba(X)[:, 1].reshape(-1, 1)

        meta_features = np.hstack([xgb_probas, mlp_probas])
        meta_probas = self.meta_clf.predict_proba(meta_features)
        return meta_probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
