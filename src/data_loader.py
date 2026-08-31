import os
import re
import numpy as np
import pandas as pd


def load_and_preprocess_data(
    file_path: str,
    sample_size: int = None,
    output_path: str = "data/processed/cleaned_loans.parquet",
) -> pd.DataFrame:
    """
    Loads raw LendingClub data, filters indeterminate loans,
    creates the binary default target, and engineers core risk features.
    """
    print(f"Reading dataset from: {file_path}")
    
    # Selected columns to optimize memory usage
    use_cols = [
        "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "issue_d", "loan_status", "purpose", "dti",
        "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "recoveries",
        "total_rec_prncp", "total_rec_int", "last_pymnt_amnt"
    ]

    # Ingest data (with optional subsample for quick iterations)
    if sample_size:
        df = pd.read_csv(file_path, usecols=use_cols, nrows=sample_size, low_memory=False)
    else:
        df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

    print(f"Initial raw records loaded: {len(df):,}")

    # 1. Target Definition (Drop non-resolved / indeterminate loans)
    default_statuses = {
        "Charged Off": 1,
        "Default": 1,
        "Does not meet the credit policy. Status:Charged Off": 1,
    }
    non_default_statuses = {
        "Fully Paid": 0,
        "Does not meet the credit policy. Status:Fully Paid": 0,
    }
    target_mapping = {**default_statuses, **non_default_statuses}

    df["is_default"] = df["loan_status"].map(target_mapping)
    df = df.dropna(subset=["is_default"]).copy()
    df["is_default"] = df["is_default"].astype(int)

    print(f"Resolved loan records retained: {len(df):,}")
    print(f"Target Distribution:\n{df['is_default'].value_counts(normalize=True)}")

    # 2. Term & Employment Length Cleaning
    # Convert ' 36 months' -> 36
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)

    # Convert emp_length: '< 1 year' -> 0, '10+ years' -> 10, etc.
    def clean_emp_length(val):
        if pd.isna(val):
            return np.nan
        if "< 1" in str(val):
            return 0.0
        match = re.search(r"\d+", str(val))
        return float(match.group()) if match else np.nan

    df["emp_length_years"] = df["emp_length"].apply(clean_emp_length)

    # 3. Clean revol_util percentage strings (e.g. '45.3%' -> 45.3)
    if df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].astype(str).str.rstrip("%").astype(float)

    # 4. Date Feature Engineering: Credit History Age in Years
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    df["credit_history_years"] = df["credit_history_years"].apply(lambda x: max(x, 0) if pd.notna(x) else np.nan)

    # 5. Financial Interaction Ratios
    # Truncate annual_inc floor to prevent division by zero
    income_safe = df["annual_inc"].apply(lambda x: max(x, 1000.0) if pd.notna(x) else np.nan)
    
    df["pti_ratio"] = (df["installment"] * 12) / income_safe
    df["revol_to_income"] = df["revol_bal"] / income_safe

    # 6. Save processed dataset (Parquet format for speed & compression)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    raw_path = "data/raw/accepted_2007_to_2018Q4.csv"
    
    # Process full dataset (or set sample_size=100_000 for faster prototyping)
    df_clean = load_and_preprocess_data(raw_path, sample_size=None)
    print("\nPreprocessed Schema Overview:")
    print(df_clean.info())