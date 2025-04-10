import os
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from data_nsl_kdd import load_nsl_kdd, one_hot_encode_features, scale_features
from stack_ensemble import StackedEnsembleModel


def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Matplotlib's imshow.

    Args:
        cm (2D array): Confusion matrix (shape [n_classes, n_classes]).
        class_names (list): Names of the classes (e.g. ["Normal", "Attack"]).
    """
    plt.figure()  # create a separate figure
    plt.imshow(cm, interpolation='nearest')  # let Matplotlib pick the default colormap
    plt.title("Confusion Matrix")
    plt.colorbar()  # add a color scale

    num_classes = len(class_names)

    # Label the axes with class names
    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)

    # Display numeric counts in each cell
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def main():
    # 1. Paths
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_dir, 'data')

    train_path = os.path.join(data_dir, "KDDTrain+.csv")
    test_path = os.path.join(data_dir, "KDDTest+.csv")

    # 2. Load & preprocess data (binary classification)
    X_train_df, y_train_df, X_test_df, y_test_df = load_nsl_kdd(
        train_path, test_path, binary=True
    )
    cat_cols = ["protocol_type", "service", "flag"]
    X_train_df, X_test_df = one_hot_encode_features(X_train_df, X_test_df, cat_cols)

    X_train_np, X_test_np = scale_features(X_train_df, X_test_df)

    # 3. Instantiate stacked ensemble (XGBoost + MLP) or any supervised model
    xgb_params = {
        "n_estimators": 50,
        "max_depth": 6,
        # 'use_label_encoder': False -> not needed in newer XGBoost
    }
    mlp_params = {
        "input_dim": X_train_np.shape[1],
        "hidden_dim": 64,
        "num_classes": 2,  # 0=normal, 1=attack
        "lr": 1e-3,
        "epochs": 5,
        "batch_size": 32
    }

    model = StackedEnsembleModel(xgb_params=xgb_params, mlp_params=mlp_params)
    print("Training the stacked ensemble...")
    model.fit(X_train_np, y_train_df)

    # 4. Evaluate on test set
    print("Evaluating the model...")
    preds = model.predict(X_test_np)
    acc = accuracy_score(y_test_df, preds)
    prec = precision_score(y_test_df, preds)
    rec = recall_score(y_test_df, preds)
    f1 = f1_score(y_test_df, preds)

    # 5. Confusion matrix
    cm = confusion_matrix(y_test_df, preds)

    print("\n=== Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 6. Plot the confusion matrix
    # For binary classification: class_names = ["Normal", "Attack"]
    plot_confusion_matrix(cm, ["Normal", "Attack"])


if __name__ == "__main__":
    main()
