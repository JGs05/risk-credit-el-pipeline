import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def calculate_ks_statistic(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Calculates the Kolmogorov-Smirnov (KS) statistic."""
    goods = y_pred_prob[y_true == 0]
    bads = y_pred_prob[y_true == 1]
    ks_stat, _ = ks_2samp(goods, bads)
    return ks_stat


def scale_pd_to_score(
    pd_series: np.ndarray,
    target_score: float = 600.0,
    target_odds: float = 50.0,
    pdo: float = 20.0,
    min_score: int = 300,
    max_score: int = 850,
) -> np.ndarray:
    """Scales default probability into standard credit scorecard points (300-850)."""
    # Prevent numerical instability for edge probabilities
    pd_clipped = np.clip(pd_series, 1e-6, 1.0 - 1e-6)
    
    # Non-default to default odds: (1 - PD) / PD
    odds = (1.0 - pd_clipped) / pd_clipped

    factor = pdo / np.log(2.0)
    offset = target_score - (factor * np.log(target_odds))

    scores = offset + (factor * np.log(odds))
    return np.clip(np.round(scores), min_score, max_score).astype(int)


def train_pd_scorecard():
    data_path = "data/processed/cleaned_loans.parquet"
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter features based on IV results and standard credit risk factors
    numeric_features = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "revol_util",
        "total_acc",
        "emp_length_years",
        "credit_history_years",
        "pti_ratio",
    ]
    categorical_features = [
        "grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "term_months",
    ]

    target = "is_default"

    # Impute missing continuous entries with column median
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())

    # Impute missing categorical entries with 'Missing'
    for col in categorical_features:
        df[col] = df[col].fillna("Missing").astype(str)

    X = df[numeric_features + categorical_features]
    y = df[target].values

    # Train/Test Split (70/30 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"Training observations: {len(X_train):,} | Test observations: {len(X_test):,}")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Calibrated L2-regularized Logistic Regression
    clf = LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    print("\nTraining Logistic Regression PD Model...")
    pipeline.fit(X_train, y_train)

    # Predict default probabilities on test set
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_prob >= 0.50).astype(int)

    # Evaluation Metrics: ROC-AUC, Gini, and KS
    auc = roc_auc_score(y_test, y_test_pred_prob)
    gini = 2 * auc - 1
    ks = calculate_ks_statistic(y_test, y_test_pred_prob)

    print("\n================ PD MODEL PERFORMANCE ================")
    print(f"ROC-AUC Score    : {auc:.4f}  (Industry Benchmark: > 0.68)")
    print(f"Gini Coefficient : {gini:.4f}  (Industry Benchmark: > 0.35)")
    print(f"KS Statistic     : {ks * 100:.2f}% (Industry Benchmark: > 30.0%)")
    print("======================================================")
    print("\nClassification Report (0.50 threshold):\n")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Apply Scorecard Scaling to the entire dataset
    print("\nGenerating credit scores (300 - 850 range)...")
    full_pd = pipeline.predict_proba(X)[:, 1]
    df["predicted_pd"] = full_pd
    df["credit_score"] = scale_pd_to_score(full_pd)

    print("\nScore Distribution Summary:")
    print(df["credit_score"].describe())

    # Save trained model and scored data
    joblib.dump(pipeline, "models/pd_logistic_regression.joblib")
    df.to_parquet("data/processed/scored_loans.parquet", index=False)
    print("\nArtifacts saved:")
    print(" - Model: models/pd_logistic_regression.joblib")
    print(" - Scored Dataset: data/processed/scored_loans.parquet")


if __name__ == "__main__":
    train_pd_scorecard()