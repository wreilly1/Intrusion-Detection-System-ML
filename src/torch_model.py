# src/torch_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # for progress bars


class IDSMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(IDSMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class MLPModel:
    """
    A wrapper for training the PyTorch MLP and producing predictions
    that are scikit-learn-like (i.e., predict_proba).
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2, lr=1e-3, epochs=10, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = IDSMLP(input_dim, hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        """
        X_train, y_train should be NumPy arrays (or array-like).
        """
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        # If y_train is a pd.Series, get the underlying values
        y_train_t = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train with TQDM progress bars
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Create a progress bar for the current epoch
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")

            for X_batch, y_batch in pbar:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Optionally, update the TQDM postfix with the current loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs} - Avg Loss: {avg_loss:.4f}")

    def predict_proba(self, X):
        """
        Return Nx2 array of probabilities for binary classification.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            batch_size = 1024
            for i in range(0, len(X_t), batch_size):
                X_batch = X_t[i:i + batch_size]
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return torch.cat([torch.tensor(a) for a in all_probs], dim=0).numpy()

    def predict(self, X):
        """
        Return predicted class labels (0 or 1).
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

