Intrusion Detection System (IDS) – Stacked Ensemble
Description:
This code provides a demonstration of a stacked-ensemble Intrusion Detection System using the NSL-KDD dataset. It combines two base learners—XGBoost (a tree-based model) and a PyTorch MLP—with a logistic regression meta-classifier to improve attack detection.

What the code does:

Data Loading & Preprocessing

Reads NSL-KDD CSV files (KDDTrain+.csv, KDDTest+.csv)

Converts categorical fields (e.g., protocol, service, flag) using one-hot encoding

Scales numerical features to aid training stability

Model Training

Fits an XGBoost classifier on the processed training set

Trains a PyTorch MLP, showing training progress (e.g., per-epoch loss)

Stacking

Collects predicted probabilities from both models

Trains a logistic regression meta-classifier on those probabilities

Produces final predictions for each sample (normal or attack)

Evaluation

Prints accuracy, precision, recall, F1-score

Displays or logs a confusion matrix for deeper insight (TP, FP, FN, TN)

This approach highlights how ensembles can leverage different strengths of tree-based and neural network models for intrusion detection.
