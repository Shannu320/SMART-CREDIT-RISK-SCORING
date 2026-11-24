#!/usr/bin/env python3
"""
Create synthetic credit risk dataset for the Smart Credit Risk Scoring project.
This generates realistic loan applicant data with default outcomes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Number of samples
n_samples = 10000

# Generate applicant data
data = {}

# --- Applicant Demographics ---
data['applicant_id'] = [f'APP_{i:06d}' for i in range(n_samples)]
data['age'] = np.random.normal(42, 12, n_samples).clip(18, 75).astype(int)
data['gender'] = np.random.choice(['M', 'F'], n_samples, p=[0.52, 0.48])
data['marital_status'] = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'],
                                           n_samples, p=[0.35, 0.55, 0.08, 0.02])
data['dependents'] = np.random.poisson(1.2, n_samples).clip(0, 6)

# --- Employment & Income ---
employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Contract', 'Unemployed', 'Retired']
emp_probs = [0.55, 0.10, 0.15, 0.08, 0.05, 0.07]
data['employment_type'] = np.random.choice(employment_types, n_samples, p=emp_probs)
data['employment_tenure_years'] = np.random.exponential(5, n_samples).clip(0, 40).round(1)

# Income depends on employment type
base_income = {
    'Full-time': 65000, 'Part-time': 28000, 'Self-employed': 55000,
    'Contract': 45000, 'Unemployed': 12000, 'Retired': 35000
}
income = np.array([np.random.lognormal(np.log(base_income[emp]), 0.4) for emp in data['employment_type']])
data['annual_income'] = income.round(-2).astype(int)

# --- Credit History ---
data['credit_score'] = np.random.normal(650, 85, n_samples).clip(300, 850).astype(int)
data['credit_history_length_years'] = np.random.exponential(8, n_samples).clip(0, 40).round(1)
data['num_credit_accounts'] = np.random.poisson(4, n_samples).clip(0, 20)
data['num_delinquent_accounts'] = np.random.poisson(0.3, n_samples).clip(0, 5)
data['past_defaults'] = np.random.poisson(0.15, n_samples).clip(0, 3)
data['bankruptcy_flag'] = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])

# --- Debt Metrics ---
data['total_debt'] = np.random.lognormal(10.5, 0.8, n_samples).round(-2).astype(int)
data['monthly_debt_payment'] = (data['total_debt'] * np.random.uniform(0.015, 0.04, n_samples)).round(-1).astype(int)
data['debt_to_income_ratio'] = (data['monthly_debt_payment'] * 12 / data['annual_income']).clip(0, 1.5).round(3)

# --- Loan Details ---
data['loan_amount'] = np.random.lognormal(10.8, 0.6, n_samples).clip(1000, 100000).round(-2).astype(int)
data['loan_term_months'] = np.random.choice([12, 24, 36, 48, 60, 72, 84], n_samples, p=[0.05, 0.15, 0.25, 0.20, 0.20, 0.10, 0.05])
loan_purposes = ['Debt Consolidation', 'Home Improvement', 'Major Purchase', 'Auto', 'Medical', 'Education', 'Business', 'Other']
data['loan_purpose'] = np.random.choice(loan_purposes, n_samples, p=[0.30, 0.15, 0.15, 0.12, 0.08, 0.07, 0.08, 0.05])
data['interest_rate'] = np.random.normal(12, 4, n_samples).clip(4, 36).round(2)

# --- Repayment History (for existing loans) ---
data['num_late_payments_30d'] = np.random.poisson(0.5, n_samples).clip(0, 12)
data['num_late_payments_60d'] = np.random.poisson(0.15, n_samples).clip(0, 6)
data['num_late_payments_90d'] = np.random.poisson(0.05, n_samples).clip(0, 4)
data['on_time_payment_pct'] = np.random.beta(8, 1, n_samples).clip(0.3, 1.0).round(3)

# --- Create DataFrame ---
df = pd.DataFrame(data)

# --- Generate Default Target based on realistic risk factors ---
# Higher risk factors increase default probability
default_prob = (
    0.02  # base rate
    + 0.15 * (df['debt_to_income_ratio'] > 0.5).astype(int)
    + 0.12 * (df['credit_score'] < 600).astype(int)
    + 0.10 * (df['past_defaults'] > 0).astype(int)
    + 0.08 * (df['bankruptcy_flag'] == 1).astype(int)
    + 0.07 * (df['employment_type'] == 'Unemployed').astype(int)
    + 0.05 * (df['employment_type'] == 'Part-time').astype(int)
    + 0.06 * (df['num_late_payments_90d'] > 0).astype(int)
    + 0.04 * (df['num_late_payments_60d'] > 0).astype(int)
    + 0.03 * (df['loan_amount'] > 50000).astype(int)
    + 0.02 * (df['age'] < 25).astype(int)
    - 0.05 * (df['credit_score'] > 750).astype(int)
    - 0.03 * (df['on_time_payment_pct'] > 0.95).astype(int)
    - 0.02 * (df['employment_tenure_years'] > 5).astype(int)
)

# Add some noise
default_prob = default_prob + np.random.normal(0, 0.03, n_samples)
default_prob = default_prob.clip(0.005, 0.85)

df['default'] = np.random.binomial(1, default_prob).astype(int)

# --- Save datasets ---
output_dir = Path('/Users/premkumarreddy/projects/credit card/data')
output_dir.mkdir(parents=True, exist_ok=True)

# Main combined dataset
df.to_csv(output_dir / 'loan_dataset.csv', index=False)

# Split into separate files as per project requirements
# Applicant data
applicant_cols = ['applicant_id', 'age', 'gender', 'marital_status', 'dependents',
                  'employment_type', 'employment_tenure_years', 'annual_income']
df[applicant_cols].to_csv(output_dir / 'applicant_data.csv', index=False)

# Loan data
loan_cols = ['applicant_id', 'loan_amount', 'loan_term_months', 'loan_purpose', 'interest_rate']
df[loan_cols].to_csv(output_dir / 'loan_data.csv', index=False)

# Repayment data
repayment_cols = ['applicant_id', 'num_late_payments_30d', 'num_late_payments_60d',
                  'num_late_payments_90d', 'on_time_payment_pct']
df[repayment_cols].to_csv(output_dir / 'repayment_data.csv', index=False)

# Credit history data
credit_cols = ['applicant_id', 'credit_score', 'credit_history_length_years',
               'num_credit_accounts', 'num_delinquent_accounts', 'past_defaults',
               'bankruptcy_flag', 'total_debt', 'monthly_debt_payment', 'debt_to_income_ratio']
df[credit_cols].to_csv(output_dir / 'credit_history_data.csv', index=False)

# Target
target_cols = ['applicant_id', 'default']
df[target_cols].to_csv(output_dir / 'target_data.csv', index=False)

print(f"Dataset created with {n_samples} samples")
print(f"Default rate: {df['default'].mean():.2%}")
print(f"\nFiles saved to {output_dir}:")
for f in output_dir.glob('*.csv'):
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Print feature info
print("\n--- Feature Summary ---")
print(f"Numeric features: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical features: {df.select_dtypes(include=['object']).columns.tolist()}")
print(f"\nClass distribution:\n{df['default'].value_counts()}")