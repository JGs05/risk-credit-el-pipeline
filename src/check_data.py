import pandas as pd

data_path = "data/raw/accepted_2007_to_2018Q4.csv"

print("Loading initial 1,000 rows for schema validation...")
sample_df = pd.read_csv(data_path, nrows=1000)

print(f"Dataset shape sample: {sample_df.shape}")
print("\nKey risk columns present:")
risk_cols = ['loan_amnt', 'term', 'int_rate', 'grade', 'loan_status', 'annual_inc', 'dti']
print(sample_df[risk_cols].head())