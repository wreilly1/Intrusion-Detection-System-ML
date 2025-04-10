# src/data_nsl_kdd.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Columns in NSL-KDD
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


def load_nsl_kdd(train_path, test_path, drop_difficulty=True, binary=True):
    # Read CSV
    train_df = pd.read_csv(train_path, names=COL_NAMES, header=None)
    test_df = pd.read_csv(test_path, names=COL_NAMES, header=None)

    # Drop difficulty if requested
    if drop_difficulty:
        train_df.drop("difficulty", axis=1, inplace=True)
        test_df.drop("difficulty", axis=1, inplace=True)

    # Separate features / labels
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Binary labeling (normal=0, attack=1) or keep multi-class
    if binary:
        y_train = y_train.apply(lambda x: 0 if x == "normal" else 1)
        y_test = y_test.apply(lambda x: 0 if x == "normal" else 1)

    return X_train, y_train, X_test, y_test


def one_hot_encode_features(X_train, X_test, categorical_cols):
    combined = pd.concat([X_train, X_test], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=categorical_cols)
    X_train_enc = combined.xs("train")
    X_test_enc = combined.xs("test")
    return X_train_enc, X_test_enc


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
