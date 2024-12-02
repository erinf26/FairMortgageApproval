import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

class HMDARobustTrainerWithoutFairness:
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 1024
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train model without fairness constraints."""
        print("\nStarting model training without fairness constraints...")
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{self.epochs}], Average Loss: {avg_loss:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        print("\nEvaluating model without fairness constraints...")
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
            y_pred_binary = (y_pred > 0.5).astype(int).reshape(-1)

        accuracy = accuracy_score(y, y_pred_binary)
        balanced_accuracy = balanced_accuracy_score(y, y_pred_binary)
        f1 = f1_score(y, y_pred_binary)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1
        }

    def calculate_demographic_parity(self, X: np.ndarray, y: np.ndarray, protected_indices: List[int]) -> float:
        """Calculate demographic parity."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
            y_pred_binary = (y_pred > 0.5).astype(int).reshape(-1)

        protected_group = X[:, protected_indices]
        non_protected_group = np.delete(X, protected_indices, axis=1)

        protected_group_pred = y_pred_binary[protected_group.any(axis=1)]
        non_protected_group_pred = y_pred_binary[~protected_group.any(axis=1)]

        protected_group_rate = np.mean(protected_group_pred)
        non_protected_group_rate = np.mean(non_protected_group_pred)

        demographic_parity = abs(protected_group_rate - non_protected_group_rate)
        return demographic_parity

    def calculate_equalized_odds(self, X: np.ndarray, y: np.ndarray, protected_indices: List[int]) -> float:
        """Calculate equalized odds."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
            y_pred_binary = (y_pred > 0.5).astype(int).reshape(-1)

        protected_group = X[:, protected_indices]
        non_protected_group = np.delete(X, protected_indices, axis=1)

        protected_group_true_positive = np.mean(y_pred_binary[protected_group.any(axis=1) & (y == 1)])
        protected_group_false_positive = np.mean(y_pred_binary[protected_group.any(axis=1) & (y == 0)])
        non_protected_group_true_positive = np.mean(y_pred_binary[~protected_group.any(axis=1) & (y == 1)])
        non_protected_group_false_positive = np.mean(y_pred_binary[~protected_group.any(axis=1) & (y == 0)])

        true_positive_diff = abs(protected_group_true_positive - non_protected_group_true_positive)
        false_positive_diff = abs(protected_group_false_positive - non_protected_group_false_positive)

        equalized_odds = max(true_positive_diff, false_positive_diff)
        return equalized_odds

# Main execution
if __name__ == "__main__":
    print("Starting HMDA ML Training Pipeline without Fairness Constraints...")
    # Load and preprocess data
    filepath = '/content/hmda_2017_ct_all-records_codes.csv'
    df = pd.read_csv(filepath)
    X, y, protected_indices = preprocess_data(df)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    # Initialize and train model
    trainer = HMDARobustTrainerWithoutFairness(
        n_features=X_train.shape[1],
        epochs=50
    )
    # Train model
    trainer.train(X_train, y_train)
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Calculate demographic parity and equalized odds
    demographic_parity = trainer.calculate_demographic_parity(X_test, y_test, protected_indices)
    equalized_odds = trainer.calculate_equalized_odds(X_test, y_test, protected_indices)
    print(f"\nDemographic Parity: {demographic_parity:.4f}")
    print(f"Equalized Odds: {equalized_odds:.4f}")
