import os
import numpy as np
import pandas as pd


def compute_expected_loss_pipeline(
    input_path: str = "data/processed/scored_loans.parquet",
    output_loan_level_path: str = "data/processed/portfolio_expected_loss.parquet",
    output_summary_dir: str = "data/processed/bi_reports",
):
    print(f"Loading scored loans dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    # 1. Loss Given Default (LGD) Modeling
    defaulted_mask = df["is_default"] == 1
    realized_recovery_rate = (
        df.loc[defaulted_mask, "recoveries"].fillna(0) / 
        df.loc[defaulted_mask, "funded_amnt"].replace(0, np.nan)
    ).clip(0.0, 1.0)

    print(f"Empirical Historical Recovery Rate on Defaults: {realized_recovery_rate.mean():.2%}")

    grade_lgd_map = {
        "A": 0.40,
        "B": 0.45,
        "C": 0.50,
        "D": 0.55,
        "E": 0.60,
        "F": 0.65,
        "G": 0.70,
    }
    df["lgd"] = df["grade"].map(grade_lgd_map).fillna(0.50)

    # 2. Exposure at Default (EAD) Modeling
    df["ead"] = df["funded_amnt"].astype(float)

    # 3. Expected Loss (EL = PD * LGD * EAD)
    df["expected_loss"] = df["predicted_pd"] * df["lgd"] * df["ead"]
    df["expected_loss_rate"] = df["expected_loss"] / df["ead"].replace(0, np.nan)

    # 4. Aggregations & Reporting
    os.makedirs(output_summary_dir, exist_ok=True)

    grade_summary = (
        df.groupby("grade", observed=False)
        .agg(
            loan_count=("id", "count"),
            total_exposure=("ead", "sum"),
            avg_pd=("predicted_pd", "mean"),
            avg_lgd=("lgd", "mean"),
            total_expected_loss=("expected_loss", "sum"),
            realized_default_rate=("is_default", "mean"),
            avg_credit_score=("credit_score", "mean"),
        )
        .reset_index()
    )
    grade_summary["el_provision_rate"] = (
        grade_summary["total_expected_loss"] / grade_summary["total_exposure"]
    )

    df["issue_year"] = pd.to_datetime(df["issue_d"]).dt.year
    vintage_summary = (
        df.groupby(["issue_year", "grade"], observed=False)
        .agg(
            total_exposure=("ead", "sum"),
            total_expected_loss=("expected_loss", "sum"),
            realized_default_rate=("is_default", "mean"),
            avg_credit_score=("credit_score", "mean"),
        )
        .reset_index()
    )

    score_bins = [300, 580, 670, 740, 800, 850]
    score_labels = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Very Good (740-799)", "Exceptional (800+)"]
    df["score_band"] = pd.cut(df["credit_score"], bins=score_bins, labels=score_labels, right=False)

    score_band_summary = (
        df.groupby("score_band", observed=False)
        .agg(
            loan_count=("id", "count"),
            total_exposure=("ead", "sum"),
            total_expected_loss=("expected_loss", "sum"),
            avg_pd=("predicted_pd", "mean"),
            default_rate=("is_default", "mean"),
        )
        .reset_index()
    )

    # 5. Export Datasets & Summaries
    print("\nSaving loan-level EL dataset...")
    df.to_parquet(output_loan_level_path, index=False)

    grade_summary.to_csv(os.path.join(output_summary_dir, "grade_risk_summary.csv"), index=False)
    vintage_summary.to_csv(os.path.join(output_summary_dir, "vintage_analysis.csv"), index=False)
    score_band_summary.to_csv(os.path.join(output_summary_dir, "score_band_summary.csv"), index=False)

    total_portfolio_exposure = df["ead"].sum()
    total_portfolio_el = df["expected_loss"].sum()
    portfolio_el_rate = (total_portfolio_el / total_portfolio_exposure) * 100

    print("\n=======================================================")
    print("           PORTFOLIO RISK & CAPITAL SUMMARY           ")
    print("=======================================================")
    print(f"Total Portfolio Loans    : {len(df):,}")
    print(f"Total Exposure at Default: ${total_portfolio_exposure:,.2f}")
    print(f"Total Expected Loss (EL) : ${total_portfolio_el:,.2f}")
    print(f"Portfolio Loss Rate (EL%): {portfolio_el_rate:.2f}%")
    print("=======================================================\n")

    print("--- Grade Breakdown ---")
    formatted_grade = grade_summary.copy()
    formatted_grade["total_exposure"] = formatted_grade["total_exposure"].map("${:,.0f}".format)
    formatted_grade["total_expected_loss"] = formatted_grade["total_expected_loss"].map("${:,.0f}".format)
    formatted_grade["avg_pd"] = formatted_grade["avg_pd"].map("{:.2%}".format)
    formatted_grade["realized_default_rate"] = formatted_grade["realized_default_rate"].map("{:.2%}".format)
    formatted_grade["el_provision_rate"] = formatted_grade["el_provision_rate"].map("{:.2%}".format)
    formatted_grade["avg_credit_score"] = formatted_grade["avg_credit_score"].round(0).astype(int)
    print(formatted_grade.to_string(index=False))


if __name__ == "__main__":
    compute_expected_loss_pipeline()