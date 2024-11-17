#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os  

class FairLendingClassifier:
    def __init__(self):
        self.non_protected = ['loan_type_name', 'loan_purpose', 'owner_occupancy',
                            'loan_amount_000s', 'applicant_income_000s',
                            'purchaser_type', 'lien_status']
        self.protected = ['county_code', 'applicant_ethnicity',
                         'applicant_race_1', 'applicant_sex']
        self.predictor = ['action_taken']
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_and_preprocess(self, filepath):
        print("Loading data...")
        df = pd.read_csv(filepath, quotechar='"', on_bad_lines='skip', low_memory=False)
        print("Selecting features...")
        feature_subset = self.non_protected + self.protected + self.predictor
        print(f"Initial shape: {df.shape}")
        print("\nMissing values before cleaning:")
        print(df[feature_subset].isnull().sum())
        df = df.dropna(axis=0, how='any', subset=feature_subset)
        print(f"\nShape after dropping NaN: {df.shape}")
        df = df[feature_subset]
        print("\nMapping action_taken to binary...")
        action_mapping = {
            1: 1, 2: 1,  # approved
            3: 0, 4: 0, 5: 0, 7: 0, 8: 0  # not approved
        }
        df['action_taken'] = df['action_taken'].map(action_mapping)
        print("\nAction taken distribution:")
        print(df['action_taken'].value_counts())
        print("\nEncoding categorical variables...")
        for col in df.select_dtypes(include=['object']).columns:
            print(f"Encoding {col}...")
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        X = df.drop('action_taken', axis=1)
        y = df['action_taken']
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y

    def train_fair_model(self, X, y, sensitive_feature='applicant_race_1',
                        constraint_type='demographic_parity'):
        print("Starting model training...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        sensitive_train = X_train[sensitive_feature]
        sensitive_test = X_test[sensitive_feature]
        
        estimator = LogisticRegression(max_iter=1000, class_weight='balanced')
        
        constraint = DemographicParity() if constraint_type == 'demographic_parity' else EqualizedOdds()
        
        fair_model = ExponentiatedGradient(
            estimator,
            constraints=constraint,
            eps=0.01
        )
        fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)
        
        y_pred = fair_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'demographic_parity': demographic_parity_difference(
                y_test, y_pred, sensitive_features=sensitive_test
            ),
            'equalized_odds': equalized_odds_difference(
                y_test, y_pred, sensitive_features=sensitive_test
            )
        }
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\nFairness Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return fair_model, metrics

    def plot_fairness_metrics(self, metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Fairness Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('fairness_metrics.png')
        plt.close()

def main():
    try:
        print("Starting program...")
        print("Initializing Fair Lending Classifier...")
        classifier = FairLendingClassifier()

        print("\nLoading and preprocessing data...")
        X, y = classifier.load_and_preprocess('hmda_2017_ct_all-records_labels.csv')

        print("\nTraining model...")
        model, metrics = classifier.train_fair_model(
            X, y, 
            sensitive_feature='applicant_race_1',
            constraint_type='demographic_parity'
        )

        print("\nSaving metrics plot...")
        classifier.plot_fairness_metrics(metrics)
        
        print("\nAnalysis complete! Check fairness_metrics.png for visualization.")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()