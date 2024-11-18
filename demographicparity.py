import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def get_race_mapping():
    """Standard HMDA race codes"""
    return {
        '1': 'American Indian/Alaska Native',
        '2': 'Asian',
        '3': 'Black',
        '4': 'Hawaiian/Pacific Islander',
        '5': 'White',
        '6': 'Information Not Provided',
        '7': 'Not Applicable'
    }

def get_preprocessed_data(filepath):
    """Load and preprocess the HMDA data with validation"""
    # Read data
    data = pd.read_csv(filepath)
    
    # Print initial data shape
    print(f"Initial data shape: {data.shape}")
    
    # Basic data cleaning
    # Only keep rows where action_taken is either 1 (approved) or 3 (denied)
    data = data[data['action_taken'].isin([1, 3])]
    
    # Convert action_taken=3 to 0 for binary classification
    data['action_taken'] = (data['action_taken'] == 1).astype(int)
    
    # Handle missing values
    numeric_columns = ['loan_amount_000s', 'applicant_income_000s']
    data = data.dropna(subset=numeric_columns + ['applicant_race_1', 'action_taken'])
    
    # Remove rows with zero or negative values in numeric columns
    for col in numeric_columns:
        data = data[data[col] > 0]
    
    # Calculate loan to income ratio
    data['loan_to_income_ratio'] = data['loan_amount_000s'] / data['applicant_income_000s']
    
    # Remove extreme outliers (outside 3 standard deviations)
    for col in numeric_columns + ['loan_to_income_ratio']:
        mean = data[col].mean()
        std = data[col].std()
        data = data[abs(data[col] - mean) <= 3 * std]
    
    print(f"Final data shape after preprocessing: {data.shape}")
    
    # Validate that we have enough data
    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing!")
        
    return data

def demographic_parity(predictions, sensitive_attributes):
    """Calculates the demographic parity difference."""
    unique_groups = np.unique(sensitive_attributes)
    positive_rates = []
    
    for group in unique_groups:
        group_mask = (sensitive_attributes == group)
        if sum(group_mask) > 0:  # Only calculate if group has samples
            positive_rate = np.mean(predictions[group_mask] == 1)
            positive_rates.append(positive_rate)
    
    return max(positive_rates) - min(positive_rates) if positive_rates else 0

def equalized_odds(predictions, true_labels, sensitive_attributes):
    """Computes TPR and FPR differences between demographic groups."""
    unique_groups = np.unique(sensitive_attributes)
    tpr_list, fpr_list = [], []
    
    for group in unique_groups:
        group_mask = (sensitive_attributes == group)
        if sum(group_mask) > 0:  # Only calculate if group has samples
            cm = confusion_matrix(true_labels[group_mask], predictions[group_mask])
            if cm.shape == (2, 2):  # Check if confusion matrix is valid
                tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0
                tpr_list.append(tpr)
                fpr_list.append(fpr)
    
    return {
        'tpr_difference': max(tpr_list) - min(tpr_list) if tpr_list else 0,
        'fpr_difference': max(fpr_list) - min(fpr_list) if fpr_list else 0
    }

def analyze_fairness_metrics(df, model, X_test, y_test, sensitive_test):
    """Analyze and visualize fairness metrics"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Distributions
    plt.subplot(3, 2, 1)
    for col in ['loan_amount_000s', 'applicant_income_000s', 'loan_to_income_ratio']:
        sns.kdeplot(data=df[col], label=col, alpha=0.5)
    plt.title('Feature Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # 2. Approval Rates by Race
    plt.subplot(3, 2, 2)
    race_mapping = get_race_mapping()
    df['derived_race'] = df['applicant_race_1'].map(lambda x: race_mapping.get(str(int(x)), 'Unknown'))
    approval_rates = df.groupby('derived_race')['action_taken'].mean().sort_values(ascending=False)
    
    approval_rates.plot(kind='bar')
    plt.title('Approval Rates by Race')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45, ha='right')

    # 3. Fairness Metrics
    plt.subplot(3, 2, 3)
    predictions = model.predict(X_test)
    
    dp_diff = demographic_parity(predictions, sensitive_test)
    eo_metrics = equalized_odds(predictions, y_test, sensitive_test)
    
    metrics = [dp_diff, eo_metrics['tpr_difference'], eo_metrics['fpr_difference']]
    metric_names = ['Demographic\nParity', 'Equal Opportunity\n(TPR Difference)', 'Predictive Equality\n(FPR Difference)']
    
    plt.bar(metric_names, metrics)
    plt.title('Fairness Metrics Comparison')
    plt.ylabel('Difference')
    plt.xticks(rotation=45, ha='right')

    # 4. Model Feature Importance
    plt.subplot(3, 2, 4)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    feature_importance.plot(x='feature', y='importance', kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    return fig

def main(filepath):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = get_preprocessed_data(filepath)
    
    # Prepare features and target
    features = ['loan_amount_000s', 'applicant_income_000s', 'loan_to_income_ratio']
    X = df[features]
    y = df['action_taken']
    sensitive = df['applicant_race_1']
    
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
    
    # Analyze and visualize
    print("Generating visualizations...")
    fig = analyze_fairness_metrics(df, model, X_test, y_test, sensitive_test)
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
