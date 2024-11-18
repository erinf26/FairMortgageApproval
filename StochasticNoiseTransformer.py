import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn

class StochasticNoiseTransformer:
    def __init__(self, noise_scale=0.1, target_shift=0.2):
        self.noise_scale = noise_scale
        self.target_shift = target_shift
        self.noise_params = None

    def generate_adaptive_noise(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        noise = np.random.normal(
            loc=self.target_shift * mean,
            scale=self.noise_scale * std,
            size=data.shape
        )
        return noise

    def fit_transform(self, data, protected_attributes=None):
        noise = self.generate_adaptive_noise(data)

        if protected_attributes is not None:
            unique_groups = np.unique(protected_attributes)
            group_noise = {}

            for group in unique_groups:
                group_mask = protected_attributes == group
                group_data = data[group_mask]
                group_noise[group] = self.generate_adaptive_noise(group_data)

            adjusted_noise = np.zeros_like(noise)
            for group in unique_groups:
                group_mask = protected_attributes == group
                adjusted_noise[group_mask] = group_noise[group]

            noise = adjusted_noise

        transformed_data = data + noise
        self.noise_params = {
            'mean': np.mean(noise, axis=0),
            'std': np.std(noise, axis=0)
        }
        return transformed_data

    def plot_distribution_shift(self, original_data, transformed_data, feature_idx=0, feature_name=None):
        plt.figure(figsize=(12, 6))
        
        sns.kdeplot(original_data[:, feature_idx], label='Original', color='blue', alpha=0.5)
        sns.kdeplot(transformed_data[:, feature_idx], label='Transformed', color='red', alpha=0.5)
        
        title = f'Distribution Shift Analysis - {feature_name}' if feature_name else 'Distribution Shift Analysis'
        plt.title(title)
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()

        w_distance = wasserstein_distance(
            original_data[:, feature_idx],
            transformed_data[:, feature_idx]
        )
        plt.text(0.05, 0.95, f'Wasserstein distance: {w_distance:.4f}',
                transform=plt.gca().transAxes)
        plt.show()
        return w_distance

def get_preprocessed_data(filepath):
    """Load and preprocess the HMDA data with validation"""
    data = pd.read_csv(filepath)
    print(f"Initial data shape: {data.shape}")
    
    data = data[data['action_taken'].isin([1, 3])]
    data['action_taken'] = (data['action_taken'] == 1).astype(int)
    
    numeric_columns = ['loan_amount_000s', 'applicant_income_000s']
    data = data.dropna(subset=numeric_columns + ['applicant_race_1', 'action_taken'])
    
    for col in numeric_columns:
        data = data[data[col] > 0]
    
    data['loan_to_income_ratio'] = data['loan_amount_000s'] / data['applicant_income_000s']
    
    for col in numeric_columns + ['loan_to_income_ratio']:
        mean = data[col].mean()
        std = data[col].std()
        data = data[abs(data[col] - mean) <= 3 * std]
    
    print(f"Final data shape after preprocessing: {data.shape}")
    
    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing!")
        
    return data

def main(filepath):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = get_preprocessed_data(filepath)
    
    # Initialize the transformer
    print("Applying stochastic noise transformation...")
    noise_transformer = StochasticNoiseTransformer(noise_scale=0.1, target_shift=0.2)
    
    # Apply transformation to the numerical features
    numerical_features = ['loan_amount_000s', 'applicant_income_000s']
    numerical_data = df[numerical_features].to_numpy()
    
    # Transform the data
    transformed_data = noise_transformer.fit_transform(
        numerical_data,
        protected_attributes=df['applicant_race_1'].values
    )
    
    # Plot distribution shifts
    print("Plotting distribution shifts...")
    for i, feature in enumerate(numerical_features):
        w_dist = noise_transformer.plot_distribution_shift(
            numerical_data,
            transformed_data,
            feature_idx=i,
            feature_name=feature
        )
        print(f"Wasserstein distance for {feature}: {w_dist}")
    
    # Update DataFrame with transformed values
    df_transformed = df.copy()
    for i, feature in enumerate(numerical_features):
        df_transformed[feature] = transformed_data[:, i]
    
    # Prepare features and target for model training
    features = numerical_features + ['loan_to_income_ratio']
    X = df_transformed[features]
    y = df_transformed['action_taken']
    sensitive = df_transformed['applicant_race_1']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Unique sensitive attribute values: {sensitive.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Analyze and visualize fairness metrics
    print("Generating fairness visualizations...")
    fig = analyze_fairness_metrics(df_transformed, model, X_test, y_test, sensitive_test)
    plt.show()
    
    # Print detailed metrics
    print("\nCalculating fairness metrics...")
    predictions = model.predict(X_test)
    print(f"Demographic Parity Difference: {demographic_parity(predictions, sensitive_test):.4f}")
    
    eo_metrics = equalized_odds(predictions, y_test, sensitive_test)
    print(f"Equal Opportunity (TPR) Difference: {eo_metrics['tpr_difference']:.4f}")
    print(f"Predictive Equality (FPR) Difference: {eo_metrics['fpr_difference']:.4f}")

if __name__ == "__main__":
    filepath = '/content/hmda_2017_ct_all-records_codes.csv'
    main(filepath)
