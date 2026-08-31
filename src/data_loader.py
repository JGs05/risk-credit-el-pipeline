import os
import duckdb
import pandas as pd

def load_and_preprocess_with_sql():
    """
    Executes the SQL ETL pipeline using DuckDB to read the raw CSV, 
    perform transformations, and write directly to a Parquet file.
    """
    print("Initializing DuckDB SQL engine...")
    conn = duckdb.connect()

    sql_file_path = "sql/clean_lending_club.sql"
    output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading SQL script from: {sql_file_path}")
    with open(sql_file_path, "r") as f:
        sql_query = f.read()

    print("Executing SQL transformations... (This may take a moment for large CSVs)")
    conn.execute(sql_query)
    
    print("SQL ETL complete: data/processed/cleaned_loans.parquet generated.")

    # Quick validation printout
    df_preview = pd.read_parquet("data/processed/cleaned_loans.parquet", columns=["is_default", "pti_ratio"])
    print(f"\nTotal records processed: {len(df_preview):,}")
    print(f"Target Distribution:\n{df_preview['is_default'].value_counts(normalize=True)}")


if __name__ == "__main__":
    load_and_preprocess_with_sql()