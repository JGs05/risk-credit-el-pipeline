# Credit Risk Modeling and Expected Loss (EL) Pipeline

An end-to-end consumer banking analytics framework and regulatory credit scorecard built on LendingClub loan data. This pipeline models Probability of Default (PD), empirical Loss Given Default (LGD), and Exposure at Default (EAD) to estimate loan-level and portfolio-level Expected Loss (EL) and capital reserve requirements adhering to Basel II/III and IFRS 9 guidelines.

---

## Business Problem & Overview

Financial institutions must accurately forecast whether a borrower will default on a credit facility to properly price credit risk, make sound underwriting decisions, and allocate regulatory capital reserves. Under the Basel internal ratings-based (IRB) framework and IFRS 9 impairment standards, banks compute credit provisions using the Expected Loss formulation:

EL = PD * LGD * EAD

This repository provides a modular, production-oriented Python pipeline that ingests millions of lending records, cleans and engineers financial metrics, runs Weight of Evidence (WoE) and Information Value (IV) variable selections, trains a calibrated Logistic Regression scorecard scaled to FICO-style credit scores (300-850), and calculates aggregate portfolio capital provisions.

---

## Pipeline Architecture

```text
Raw LendingClub Data (.csv)
            |
            v
+--------------------------------------------------------+
| 1. Data Cleaning & Feature Engineering (data_loader.py)|
|    - Filter unresolved/indeterminate loan records      |
|    - Define binary default target (is_default)         |
|    - Engineer PTI, revol-to-income, credit line age    |
+--------------------------------------------------------+
            |
            v
+--------------------------------------------------------+
| 2. Feature Selection & Analysis (woe_transformer.py)   |
|    - Quantile binning for non-linear relations         |
|    - Compute Weight of Evidence (WoE) per bucket       |
|    - Rank predictive power via Information Value (IV)  |
+--------------------------------------------------------+
            |
            v
+--------------------------------------------------------+
| 3. Scorecard Modeling & Calibration (pd_scorecard.py)  |
|    - Train balanced L2 Logistic Regression             |
|    - Validate discrimination: ROC-AUC, Gini, KS-Stat   |
|    - Scale log-odds into 300-850 credit scores         |
+--------------------------------------------------------+
            |
            v
+--------------------------------------------------------+
| 4. Capital Reserve & Expected Loss (expected_loss.py)  |
|    - Empirical historical recovery rates (LGD)         |
|    - Individual loan balance evaluation (EAD)          |
|    - EL = PD * LGD * EAD calculation                   |
|    - Generate BI aggregations (Grade, Vintage, Bands)  |
+--------------------------------------------------------+
            |
            v
Executive BI Dashboards (Power BI / Tableau)

Tech Stack & Core Libraries

Language: Python 3.10+
Data Manipulation & Storage: Pandas, NumPy, PyArrow (Parquet)
Machine Learning & Risk Statistics: Scikit-Learn, SciPy (Two-sample KS Test), Statsmodels
Development & Version Control: VS Code, Git, GitHub
Business Intelligence & Visualization: Tableau / Power BI compatible CSV data marts

Methodology & Formulations

1. Target Definition

Loans are categorized based on terminal resolution:
Default / High Risk (Y = 1): Charged Off, Default, Does not meet credit policy: Charged Off
Performing / Good (Y = 0): Fully Paid, Does not meet credit policy: Fully Paid
Indeterminate loans (Current, In Grace Period) are dropped to prevent point-in-time target contamination.

2. Weight of Evidence (WoE) & Information Value (IV)

For continuous and categorical variables:
WoE = ln(% Non-Defaults / % Defaults)
IV = Sum((% Non-Defaults - % Defaults) * WoE)
Features with IV >= 0.02 are selected for scorecard inclusion.

3. FICO-Style Credit Score Scaling

The model translates individual default log-odds into standard credit points:
Factor = PDO / ln(2)
Offset = Target Score - (Factor * ln(Target Odds))
Score = Offset + Factor * ln((1 - PD) / PD)
Parameters: Baseline score = 600 at 50:1 odds, PDO (Points to Double Odds) = 20, Score Range = [300, 850].

4. Loss Given Default (LGD) & Exposure at Default (EAD)

LGD: Calibrated using empirical recovery rates mapped monotonically by credit grade tier (Grade A: 40% to Grade G: 70%).
EAD: Evaluated as total funded facility amount at point-of-underwriting risk exposure.

Project Structure
credit-risk-el-pipeline/
|-- data/
|   |-- raw/                             # Source dataset (gitignored)
|   |   `-- accepted_2007_to_2018Q4.csv
|   `-- processed/                       # Processed Parquet data files
|       |-- cleaned_loans.parquet
|       |-- scored_loans.parquet
|       |-- portfolio_expected_loss.parquet
|       `-- bi_reports/                  # Ready-to-import CSVs for Tableau/Power BI
|           |-- grade_risk_summary.csv
|           |-- vintage_analysis.csv
|           `-- score_band_summary.csv
|-- models/                              # Persisted pipeline models
|   `-- pd_logistic_regression.joblib
|-- src/
|   |-- check_data.py                    # Data ingestion sanity check
|   |-- data_loader.py                   # Data cleaning, target definition, ratio engineering
|   |-- woe_transformer.py               # WoE binning and IV ranking engine
|   |-- pd_scorecard.py                  # PD Logistic Regression & FICO score calibration
|   `-- expected_loss.py                 # LGD, EAD, and portfolio expected loss pipeline
|-- .gitignore
|-- requirements.txt
`-- README.md

Installation & Setup Instructions

1. Clone the Repository

Bash
git clone [https://github.com/JGs05/credit-risk-el-pipeline.git](https://github.com/JGs05/credit-risk-el-pipeline.git)
cd credit-risk-el-pipeline

2. Create and Activate Virtual EnvironmentWindows 
(PowerShell):PowerShellSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m venv venv
.\venv\Scripts\Activate.ps1

macOS / Linux:Bashpython3 -m venv venv
source venv/bin/activate

3. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt

4. Download the LendingClub DatasetEnsure your kaggle.json token is configured in ~/.kaggle/ (or C:\Users\<Username>\.kaggle\), then run:Bashkaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
Execution PipelineRun the scripts in sequential order:
Bash# 
Step 1: Preprocess raw data & engineer financial ratios
python src/data_loader.py

# Step 2: Calculate WoE and Information Value summary
python src/woe_transformer.py

# Step 3: Train PD classifier and calibrate credit scores
python src/pd_scorecard.py

# Step 4: Compute LGD, EAD, Expected Loss, and generate BI reports
python src/expected_loss.py

Key Performance Metrics & Benchmark Targets

Metric,Model Result,Regulatory Benchmark,Assessment
ROC-AUC,~0.71,> 0.68,Strong discrimination power across borrower classes
Gini Coefficient,~0.42,> 0.35,Solid classification separation
Kolmogorov-Smirnov (KS),~32.5%,> 30.0%,Exceeds regulatory cutoff criteria

Business Intelligence (BI) Outputs
The pipeline exports three production-ready summary datasets into data/processed/bi_reports/:

grade_risk_summary.csv: Loan counts, total exposure, average PD, provision rates, and expected loss dollar figures segmented across loan tiers A through G.

vintage_analysis.csv: Longitudinal default rate curves and loss migrations across origination vintages (2007-2018).

score_band_summary.csv: Exposure and provisioning broken down across standard FICO risk bands (Poor to Exceptional).