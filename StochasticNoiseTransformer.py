import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

class StochasticNoiseTransformer:
    def __init__(self, noise_scale=0.1, target_shift=0.2):
        """
        Initialize the transformer with noise parameters

        Args:
            noise_scale (float): Scale of the stochastic noise
            target_shift (float): Desired amount of distribution shift
        """
        self.noise_scale = noise_scale
        self.target_shift = target_shift
        self.noise_params = None

    def generate_adaptive_noise(self, data):
        """
        Generate adaptive noise based on data distribution
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        noise = np.random.normal(
            loc=self.target_shift * mean,
            scale=self.noise_scale * std,
            size=data.shape
        )
        return noise

    def fit_transform(self, data, protected_attributes=None):
        """
        Fit the transformer to the data and apply the transformation

        Args:
            data: Input data array
            protected_attributes: Protected attribute indicators
        """
        # Generate initial noise
        noise = self.generate_adaptive_noise(data)

        # If protected attributes are provided, adjust noise to maintain fairness
        if protected_attributes is not None:
            unique_groups = np.unique(protected_attributes)
            group_noise = {}

            # Calculate group-specific noise adjustments
            for group in unique_groups:
                group_mask = protected_attributes == group
                group_data = data[group_mask]
                group_noise[group] = self.generate_adaptive_noise(group_data)

            # Apply group-specific noise
            adjusted_noise = np.zeros_like(noise)
            for group in unique_groups:
                group_mask = protected_attributes == group
                adjusted_noise[group_mask] = group_noise[group]

            noise = adjusted_noise

        # Apply noise to shift distribution
        transformed_data = data + noise

        # Store noise parameters for later use
        self.noise_params = {
            'mean': np.mean(noise, axis=0),
            'std': np.std(noise, axis=0)
        }

        return transformed_data

    def plot_distribution_shift(self, original_data, transformed_data, feature_idx=0):
        """
        Plot the original and transformed distributions for a specific feature

        Args:
            original_data: Original input data
            transformed_data: Transformed data
            feature_idx: Index of the feature to plot
        """
        plt.figure(figsize=(12, 6))

        # Plot original distribution
        sns.kdeplot(original_data[:, feature_idx], label='Original', color='blue', alpha=0.5)

        # Plot transformed distribution
        sns.kdeplot(transformed_data[:, feature_idx], label='Transformed', color='red', alpha=0.5)

        plt.title('Distribution Shift Analysis')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()

        # Calculate Wasserstein distance
        w_distance = wasserstein_distance(
            original_data[:, feature_idx],
            transformed_data[:, feature_idx]
        )
        plt.text(0.05, 0.95, f'Wasserstein distance: {w_distance:.4f}',
                transform=plt.gca().transAxes)

        plt.show()

        return w_distance

# Load and preprocess data as before
filepath = '/content/hmda_2017_ct_all-records_codes.csv'
df = pd.read_csv(filepath)

# Initialize the transformer
noise_transformer = StochasticNoiseTransformer(noise_scale=0.1, target_shift=0.2)

# Apply transformation to the numerical features
numerical_features = ['loan_amount_000s', 'applicant_income_000s']
numerical_data = df[numerical_features].to_numpy()

# Transform the data
transformed_data = noise_transformer.fit_transform(
    numerical_data,
    protected_attributes=df['applicant_race_1'].values
)

# Create a comparison plot for each numerical feature
for i, feature in enumerate(numerical_features):
    w_dist = noise_transformer.plot_distribution_shift(
        numerical_data,
        transformed_data,
        feature_idx=i
    )
    print(f"Wasserstein distance for {feature}: {w_dist}")

# Update the DataFrame with transformed values
df_transformed = df.copy()
for i, feature in enumerate(numerical_features):
    df_transformed[feature] = transformed_data[:, i]

# Proceed with model training using df_transformed instead of original df
