#!/usr/bin/env python3

# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame

# define feature sets
non_protected = [
    'loan_type_name', 'loan_purpose', 'owner_occupancy', 'loan_amount_000s',
    'applicant_income_000s', 'purchaser_type', 'lien_status'
]
protected = ['county_code', 'applicant_ethnicity', 'applicant_race_1', 'applicant_sex']
predictor = ['action_taken']
feature_subset = non_protected + protected + predictor

def preprocess_data(df):
    """Preprocess data by removing rows with NAs in relevant columns."""
    df = df.dropna(axis=0, how='any', subset=feature_subset)
    df = df[feature_subset]
    return df

def main():
    # load dataset
    df = pd.read_csv('hmda_2017_ct_all-records_labels.csv', quotechar='"', on_bad_lines='skip')
    
    # clean the data
    cleaned = preprocess_data(df)
    
    # print original unique values
    print("Original unique values in action_taken column:", cleaned['action_taken'].unique())
    
    # define mapping for action_taken according to website
    action_mapping = {
        1: 1,  # loan originated
        2: 1,  # application approved but not accepted
        3: 0,  # application denied
        4: 0,  # application withdrawn by applicant
        5: 0,  # file closed for incompleteness
        7: 0,  # preapproval request denied
        8: 0   # preapproval request approved but not accepted
    }
    
    # apply mapping
    cleaned['action_taken'] = cleaned['action_taken'].map(action_mapping)
    print("\nAfter mapping:")
    print(cleaned['action_taken'].value_counts())
    
    # calculate and display approval distributions
    # by race:
    race_distribution = cleaned.groupby('applicant_race_1')['action_taken'].agg(['count', 'mean'])
    race_distribution.columns = ['Total', 'Approval_Rate']
    print("\nApproval Distribution by Race:")
    print(race_distribution)
    
    # by ethnicity:
    ethnicity_distribution = cleaned.groupby('applicant_ethnicity')['action_taken'].agg(['count', 'mean'])
    ethnicity_distribution.columns = ['Total', 'Approval_Rate']
    print("\nApproval Distribution by Ethnicity:")
    print(ethnicity_distribution)
    
    # by sex:
    sex_distribution = cleaned.groupby('applicant_sex')['action_taken'].agg(['count', 'mean'])
    sex_distribution.columns = ['Total', 'Approval_Rate']
    print("\nApproval Distribution by Sex:")
    print(sex_distribution)
    
    # create plots
    # race distribution plot
    plt.figure(figsize=(12, 6))
    race_distribution['Approval_Rate'].plot(kind='bar')
    plt.title('Approval Rates by Race')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('race_distribution.png')
    plt.close()
    
    # ethnicity distribution plot
    plt.figure(figsize=(8, 6))
    ethnicity_distribution['Approval_Rate'].plot(kind='bar')
    plt.title('Approval Rates by Ethnicity')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ethnicity_distribution.png')
    plt.close()
    
    # sex distribution plot
    plt.figure(figsize=(8, 6))
    sex_distribution['Approval_Rate'].plot(kind='bar')
    plt.title('Approval Rates by Sex')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sex_distribution.png')
    plt.close()
    
    # final data checks
    print("\nValue counts in action_taken:")
    print(cleaned['action_taken'].value_counts())
    
    # prepare final datasets
    cleaned = cleaned.dropna(subset=['action_taken'])
    cleaned_non_protected = cleaned[non_protected]
    cleaned_protected = cleaned[protected]
    target = cleaned['action_taken']
    
    print("\nAny NaN values in target:", target.isna().sum())
    print("\nFirst few rows of cleaned data:")
    print(cleaned.head())

if __name__ == "__main__":
    main()