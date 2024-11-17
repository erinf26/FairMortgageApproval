import numpy as np
import pandas as pd
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

class SVMPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, probability=True)

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)

class SVMFairLendingClassifier(FairLendingClassifier):
    def __init__(self, kernel='rbf'):
        super().__init__()
        self.kernel = kernel

    def train_svm_fair_model(self, X, y, sensitive_feature='applicant_race_1'):
        print("\nStarting SVM training with fair constraints...")


        svm_estimator = SVMPipeline(kernel=self.kernel)

        # split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # sensitive features
        sensitive_train = X_train[sensitive_feature]
        sensitive_test = X_test[sensitive_feature]

        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print("\nClass distribution in training set:")
        print(pd.Series(y_train).value_counts(normalize=True))

        # train fair model with SVM
        print("\nTraining fair SVM model...")
        fair_model = ExponentiatedGradient(
            estimator=svm_estimator,
            constraints=DemographicParity(),
            eps=0.01
        )

        try:
            fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise

        # predictions
        print("\nMaking predictions...")
        y_pred = fair_model.predict(X_test)

        # metrics
        print("\nCalculating metrics...")
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'demographic_parity': demographic_parity_difference(
                y_test, y_pred, sensitive_features=sensitive_test
            ),
            'equalized_odds': equalized_odds_difference(
                y_test, y_pred, sensitive_features=sensitive_test
            )
        }

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Print fairness metrics
        print("\nFairness Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        return fair_model, metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

if __name__ == "__main__":
    try:
        print("Initializing SVM classifier...")
        svm_classifier = SVMFairLendingClassifier(kernel='rbf')

        print("\nLoading and preprocessing data...")
        X, y = svm_classifier.load_and_preprocess('hmda_2017_ct_all-records_labels.csv')

        print("\nTraining SVM model...")
        model, metrics = svm_classifier.train_svm_fair_model(X, y)

        print("\nPlotting fairness metrics...")
        svm_classifier.plot_fairness_metrics(metrics)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()