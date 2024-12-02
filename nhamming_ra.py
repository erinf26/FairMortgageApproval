# calculates the demographic parity, equalized odds, and the accuracy (best results)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from itertools import combinations

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Preprocess HMDA data with proper error checking"""
    print("Preprocessing HMDA data...")
    
    # Print initial data info
    print(f"Initial data shape: {df.shape}")
    print("\nAvailable columns:", df.columns.tolist())
    
    # Define feature categories
    non_protected = ['loan_type', 'loan_purpose', 'owner_occupancy', 'loan_amount_000s',
                     'applicant_income_000s', 'purchaser_type', 'lien_status']
    protected = ['county_code', 'applicant_ethnicity', 'applicant_race_1', 'applicant_sex']
    predictor = ['action_taken']
    
    # Verify all required columns exist
    missing_cols = [col for col in non_protected + protected + predictor if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 1. Initial Data Cleaning
    df = df.copy()
    df = df.dropna(subset=non_protected + protected + predictor)
    print(f"\nShape after dropping NA: {df.shape}")
    
    # 2. Feature Engineering
    if 'loan_amount_000s' in df.columns and 'applicant_income_000s' in df.columns:
        df['loan_to_income_ratio'] = (df['loan_amount_000s'] / 
                                    df['applicant_income_000s']).replace([np.inf, -np.inf], 0)
        non_protected.append('loan_to_income_ratio')
    
    # Print sample of data before encoding
    print("\nSample of data before encoding:")
    print(df[non_protected + protected + predictor].head())
    
    # 3. Handle categorical variables
    feature_frames = []
    
    # Process protected attributes
    protected_encoded = []
    for col in protected:
        le = LabelEncoder()
        encoded_col = f"{col}_encoded"
        df[encoded_col] = le.fit_transform(df[col].astype(str))
        protected_encoded.append(encoded_col)
        feature_frames.append(df[[encoded_col]])
    
    # Process non-protected attributes
    categorical_non_protected = [col for col in non_protected if df[col].dtype == 'object']
    numerical_non_protected = [col for col in non_protected if df[col].dtype != 'object']
    
    # Handle numerical features
    if numerical_non_protected:
        scaler = StandardScaler()
        numerical_scaled = pd.DataFrame(
            scaler.fit_transform(df[numerical_non_protected]),
            columns=numerical_non_protected,
            index=df.index
        )
        feature_frames.append(numerical_scaled)
    
    # Handle categorical features
    if categorical_non_protected:
        categorical_dummies = pd.get_dummies(
            df[categorical_non_protected],
            prefix=categorical_non_protected,
            dummy_na=True
        )
        feature_frames.append(categorical_dummies)
    
    # Print debug information
    print("\nFeature frames to concatenate:")
    for i, frame in enumerate(feature_frames):
        print(f"Frame {i} shape: {frame.shape}")
    
    # Combine all features
    if not feature_frames:
        raise ValueError("No features available after preprocessing")
    
    final_df = pd.concat(feature_frames, axis=1)
    
    # Prepare final arrays
    X_features = final_df.values.astype(np.float32)
    y = (df[predictor].values == 1).astype(np.float32).ravel()
    
    # Get indices of protected attributes
    protected_idx = [final_df.columns.get_loc(col) for col in protected_encoded]
    
    print("\nFinal preprocessing results:")
    print(f"Features shape: {X_features.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Protected attribute indices: {protected_idx}")
    print("\nSample of preprocessed features:")
    print(final_df.head())
    
    return X_features, y, protected_idx



class HammingPerturbation:
    def __init__(self, n_hamming: int, protected_idx: List[int]):
        self.n_hamming = n_hamming
        self.protected_idx = protected_idx
        
    def generate_hamming_perturbations(self, X: np.ndarray) -> List[np.ndarray]:
        """Generate n-Hamming distance perturbations for protected attributes."""
        perturbations = []
        
        for n in range(1, self.n_hamming + 1):
            feature_combinations = list(combinations(self.protected_idx, n))
            
            for feature_set in feature_combinations:
                X_perturbed = X.copy()
                
                for feature_idx in feature_set:
                    feature_values = X[:, feature_idx]
                    feature_std = np.std(feature_values)
                    unique_vals = np.unique(feature_values)
                    
                    if len(unique_vals) <= 5:  # Categorical
                        perturbation = np.random.choice(unique_vals, size=X.shape[0])
                        X_perturbed[:, feature_idx] = perturbation
                    else:  # Continuous
                        perturbation = np.random.normal(0, feature_std * 0.1, size=X.shape[0])
                        X_perturbed[:, feature_idx] = np.clip(
                            X_perturbed[:, feature_idx] + perturbation,
                            np.min(feature_values),
                            np.max(feature_values)
                        )
                
                perturbations.append(X_perturbed)
        
        return perturbations

class HMDARobustTrainer:
    def __init__(
        self,
        n_features: int,
        protected_idx: List[int],
        n_hamming: int = 3,
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
        
        self.protected_idx = protected_idx
        self.n_hamming = n_hamming
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hamming_perturbation = HammingPerturbation(n_hamming, protected_idx)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train model with n-Hamming distance robustness."""
        print("\nStarting model training...")
        print(f"Training with n-Hamming distance: {self.n_hamming}")
        
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
                
                # Forward pass
                y_pred = self.model(batch_X)
                main_loss = criterion(y_pred, batch_y)
                
                # Generate Hamming perturbations
                X_np = batch_X.cpu().detach().numpy()
                perturbed_versions = self.hamming_perturbation.generate_hamming_perturbations(X_np)
                
                # Compute robustness loss
                robustness_losses = []
                for X_perturbed in perturbed_versions:
                    X_perturbed_tensor = torch.FloatTensor(X_perturbed).to(self.device)
                    y_pred_perturbed = self.model(X_perturbed_tensor)
                    rob_loss = torch.mean(torch.abs(y_pred - y_pred_perturbed))
                    robustness_losses.append(rob_loss)
                
                hamming_loss = torch.mean(torch.stack(robustness_losses)) if robustness_losses else torch.tensor(0.0).to(self.device)
                
                # Total loss
                loss = main_loss + 0.1 * hamming_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Batch [{batch_idx}/{len(dataloader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Hamming Loss: {hamming_loss.item():.4f}")
        
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Average Loss: {avg_loss:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance with accuracy, demographic parity, and equalized odds.
        """
        print("\nEvaluating model...")
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        
        y_pred_binary = (y_pred > 0.5).astype(int).reshape(-1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_binary == y)
        
        # Calculate fairness metrics
        dp_scores = []
        eo_scores = []
        
        for idx in self.protected_idx:
            protected_vals = X[:, idx]
            protected_mask = protected_vals > np.median(protected_vals)
            
            # Demographic Parity
            prob_pos_protected = np.mean(y_pred_binary[protected_mask])
            prob_pos_unprotected = np.mean(y_pred_binary[~protected_mask])
            dp_diff = abs(prob_pos_protected - prob_pos_unprotected)
            dp_scores.append(dp_diff)
            
            # Equalized Odds
            # True Positive Rate
            tpr_protected = np.mean(y_pred_binary[(protected_mask) & (y == 1)] == 1)
            tpr_unprotected = np.mean(y_pred_binary[(~protected_mask) & (y == 1)] == 1)
            
            # False Positive Rate
            fpr_protected = np.mean(y_pred_binary[(protected_mask) & (y == 0)] == 1)
            fpr_unprotected = np.mean(y_pred_binary[(~protected_mask) & (y == 0)] == 1)
            
            # Calculate equalized odds as average of TPR and FPR differences
            eo_diff = (abs(tpr_protected - tpr_unprotected) + 
                      abs(fpr_protected - fpr_unprotected)) / 2
            eo_scores.append(eo_diff)
            
            # Print detailed metrics for each protected attribute
            print(f"\nProtected attribute {idx} metrics:")
            print(f"Demographic Parity difference: {dp_diff:.4f}")
            print(f"True Positive Rate - Protected: {tpr_protected:.4f}, Unprotected: {tpr_unprotected:.4f}")
            print(f"False Positive Rate - Protected: {fpr_protected:.4f}, Unprotected: {fpr_unprotected:.4f}")
            print(f"Equalized Odds difference: {eo_diff:.4f}")
        
        return {
            'accuracy': accuracy,
            'demographic_parity': np.mean(dp_scores),
            'equalized_odds': np.mean(eo_scores)
        }
# Main execution
if __name__ == "__main__":
    print("Starting HMDA Fair ML Training Pipeline...")
    
    # Load and preprocess data
    filepath = '/content/hmda_2017_ct_all-records_codes.csv'
    df = pd.read_csv(filepath)
    X, y, protected_idx = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and train model
    trainer = HMDARobustTrainer(
        n_features=X_train.shape[1],
        protected_idx=protected_idx,
        n_hamming=3,
        epochs=50
    )
    
    # Train model
    trainer.train(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
