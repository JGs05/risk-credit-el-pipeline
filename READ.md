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
