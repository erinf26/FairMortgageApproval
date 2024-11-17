import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from logistic_regression import FairLendingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

class PCAFairLendingClassifier(FairLendingClassifier):
    def __init__(self):
        super().__init__()
        self.pca = PCA(n_components=2)

    def apply_pca(self, X, protected_features):
        X_protected = X[protected_features]
        X_non_protected = X.drop(columns=protected_features)
        X_pca = self.pca.fit_transform(X_non_protected)

        # df with PCA results
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=['PC1', 'PC2'],
            index=X.index
        )

        for col in protected_features:
            X_pca_df[col] = X_protected[col]

        explained_variance = self.pca.explained_variance_ratio_

        print("\nPCA Explained Variance Ratio:", explained_variance)
        print(f"Total variance explained: {sum(explained_variance):.2%}")

        return X_pca_df, explained_variance

    def plot_pca_results(self, X_pca, y, sensitive_feature_values):
        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        scatter = plt.scatter(
            X_pca['PC1'],
            X_pca['PC2'],
            c=y,
            cmap='viridis',
            alpha=0.5
        )
        plt.title('PCA Results by Approval Status')
        plt.xlabel(f'First Principal Component (Var: {self.pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Var: {self.pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, label='Approval Status')

        plt.subplot(122)
        scatter = plt.scatter(
            X_pca['PC1'],
            X_pca['PC2'],
            c=sensitive_feature_values,
            cmap='Set3',
            alpha=0.5
        )
        plt.title('PCA Results by Protected Attribute')
        plt.xlabel(f'First Principal Component (Var: {self.pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Var: {self.pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, label='Protected Attribute')

        plt.tight_layout()
        plt.show()

    def train_pca_fair_model(self, X, y, sensitive_feature='applicant_race_1'):
        print("\nStarting PCA analysis...")

        sensitive_feature_values = X[sensitive_feature].copy()

        # apply PCA while preserving protected features
        X_pca, explained_variance = self.apply_pca(X, self.protected)

        print("\nShape of PCA-transformed data:", X_pca.shape)
        print("Columns in PCA data:", X_pca.columns.tolist())

        # train fair model using PCA features
        print("\nTraining fair model on PCA-transformed data...")
        model, metrics = self.train_fair_model(
            X_pca,
            y,
            sensitive_feature=sensitive_feature
        )

        # plot
        print("\nPlotting PCA results...")
        self.plot_pca_results(X_pca, y, sensitive_feature_values)

        # component weights
        print("\nPCA Component Weights:")
        feature_weights = pd.DataFrame(
            self.pca.components_,
            columns=X.drop(columns=self.protected).columns,
            index=['PC1', 'PC2']
        )
        print(feature_weights)

        return model, metrics, explained_variance


if __name__ == "__main__":
    try:
        print("Initializing PCA classifier...")
        pca_classifier = PCAFairLendingClassifier()

        print("\nLoading and preprocessing data...")
        X, y = pca_classifier.load_and_preprocess('hmda_2017_ct_all-records_labels.csv')

        print("\nTraining PCA model...")
        model, metrics, variance = pca_classifier.train_pca_fair_model(X, y)

        print("\nPrinting metrics...")
        pca_classifier.plot_fairness_metrics(metrics)
        print(f"\nExplained variance ratio: {variance}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()