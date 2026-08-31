import numpy as np
import pandas as pd


class WoETransformer:
    def __init__(self, target_col: str, max_bins: int = 5):
        self.target_col = target_col
        self.max_bins = max_bins
        self.woe_dict = {}
        self.iv_summary = {}

    def _calculate_woe_iv(self, df: pd.DataFrame, feature: str, is_numeric: bool):
        total_good = (df[self.target_col] == 0).sum()
        total_bad = (df[self.target_col] == 1).sum()

        if is_numeric:
            # Quantile binning for numeric features with handling for duplicates
            try:
                binned = pd.qcut(df[feature], q=self.max_bins, duplicates="drop")
            except ValueError:
                binned = pd.cut(df[feature], bins=self.max_bins)
            temp = pd.DataFrame({"bin": binned.astype(str), "target": df[self.target_col]})
            temp["bin"] = temp["bin"].fillna("Missing")
        else:
            temp = pd.DataFrame({"bin": df[feature].fillna("Missing").astype(str), "target": df[self.target_col]})

        grouped = temp.groupby("bin", observed=False)["target"].agg(["count", "sum"]).reset_index()
        grouped.rename(columns={"sum": "bads", "count": "total"}, inplace=True)
        grouped["goods"] = grouped["total"] - grouped["bads"]

        # Laplace-style smoothing to prevent division by zero or log(0)
        grouped["goods"] = grouped["goods"].replace(0, 0.5)
        grouped["bads"] = grouped["bads"].replace(0, 0.5)

        grouped["dist_good"] = grouped["goods"] / total_good
        grouped["dist_bad"] = grouped["bads"] / total_bad

        grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
        grouped["iv_bin"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

        total_iv = grouped["iv_bin"].sum()
        woe_map = dict(zip(grouped["bin"], grouped["woe"]))

        return woe_map, total_iv

    def fit(self, df: pd.DataFrame, continuous_cols: list, categorical_cols: list):
        print("Computing WoE and Information Value (IV)...")
        for col in continuous_cols:
            woe_map, iv = self._calculate_woe_iv(df, col, is_numeric=True)
            self.woe_dict[col] = {"type": "numeric", "map": woe_map}
            self.iv_summary[col] = iv

        for col in categorical_cols:
            woe_map, iv = self._calculate_woe_iv(df, col, is_numeric=False)
            self.woe_dict[col] = {"type": "categorical", "map": woe_map}
            self.iv_summary[col] = iv

        iv_df = pd.DataFrame(list(self.iv_summary.items()), columns=["Feature", "IV"]).sort_values(
            by="IV", ascending=False
        )
        return iv_df


if __name__ == "__main__":
    data_path = "data/processed/cleaned_loans.parquet"
    print(f"Loading cleaned data from {data_path}...")
    df = pd.read_parquet(data_path)

    continuous_features = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "revol_bal", "revol_util", "total_acc",
        "emp_length_years", "credit_history_years", "pti_ratio", "revol_to_income"
    ]

    categorical_features = [
        "grade", "sub_grade", "home_ownership",
        "verification_status", "purpose", "term_months"
    ]

    transformer = WoETransformer(target_col="is_default", max_bins=5)
    iv_summary_df = transformer.fit(df, continuous_features, categorical_features)

    print("\n--- Information Value (IV) Summary ---")
    print(iv_summary_df.to_string(index=False))

    # Save summary
    iv_summary_df.to_csv("data/processed/iv_summary.csv", index=False)
    print("\nIV Summary exported to data/processed/iv_summary.csv")