-- Read directly from the raw CSV, filter, engineer features, and export to Parquet
COPY (
    WITH raw_filtered AS (
        SELECT
            id,
            loan_amnt,
            funded_amnt,
            term,
            int_rate,
            installment,
            grade,
            sub_grade,
            emp_length,
            home_ownership,
            annual_inc,
            verification_status,
            issue_d,
            purpose,
            dti,
            delinq_2yrs,
            earliest_cr_line,
            inq_last_6mths,
            open_acc,
            pub_rec,
            revol_bal,
            revol_util,
            total_acc,
            recoveries,
            total_rec_prncp,
            -- Binary Default Target
            CASE 
                WHEN loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off') THEN 1
                WHEN loan_status IN ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid') THEN 0
                ELSE NULL 
            END AS is_default
        FROM read_csv_auto('data/raw/accepted_2007_to_2018Q4.csv')
        WHERE loan_status NOT IN ('Current', 'In Grace Period', 'Late (16-30 days)')
    )
    SELECT 
        *,
        -- Extract numeric months from term
        CAST(regexp_extract(term, '\d+') AS INTEGER) AS term_months,
        
        -- Clean emp_length to numeric
        CASE
            WHEN emp_length LIKE '< 1%' THEN 0.0
            WHEN emp_length IS NULL THEN NULL
            ELSE CAST(regexp_extract(emp_length, '\d+') AS FLOAT)
        END AS emp_length_years,

        -- Financial Ratios
        (installment * 12.0) / GREATEST(annual_inc, 1000.0) AS pti_ratio,
        revol_bal / GREATEST(annual_inc, 1000.0) AS revol_to_income
    FROM raw_filtered
    WHERE is_default IS NOT NULL
) TO 'data/processed/cleaned_loans.parquet' (FORMAT PARQUET);