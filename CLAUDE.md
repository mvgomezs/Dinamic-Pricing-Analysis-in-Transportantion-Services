# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical analysis of a dynamic pricing dataset for a ride-hailing transportation service. The goal is to explain how operational variables (ride duration, number of riders, customer loyalty) relate to **Average Ratings** using inferential statistical methods.

- **Dataset**: `Data/Dataset_dynamic_pricing.csv` — 1000 observations, 10 variables
- **Language**: Python (developed in Google Colab)
- **Notebooks**: Two Jupyter notebooks covering different phases of the analysis

## Notebooks

### `Dinamic_pricing_1.ipynb` — Inferential Statistics
Covers:
- Central Limit Theorem simulation (1000 samples, n=30) on `Expected_Ride_Duration`
- 95% Confidence intervals for mean (t-Student), proportion of excellent ratings (Z), and variance (chi-squared)
- Welch t-test: Gold vs Silver loyalty group duration comparison
- Z-test for proportions: Economy vs Premium vehicle ratings
- Fisher F-test: variance equality between loyalty groups
- Chi-squared independence test: `Customer_Loyalty_Status` × `Vehicle_Type`
- Statistical power analysis (Cohen d ≈ 0.093, power ≈ 22.9%)

### `Dinamic_pricing_2.ipynb` — Multiple Linear Regression
Covers:
- OLS model: `Average_Ratings ~ Number_of_Riders + Expected_Ride_Duration + Customer_Loyalty_Status`
- `Customer_Loyalty_Status` encoded as dummies (Gold = reference category)
- Multicollinearity diagnostics (VIF, correlation matrix)
- Outlier detection (studentized residuals, leverage, Cook's distance)
- Assumption checking: normality (Shapiro-Wilk), homoscedasticity (Breusch-Pagan), autocorrelation (Durbin-Watson)
- Model improvements: log transformation, Box-Cox transformation, robust RLM (Huber), bootstrap confidence intervals

## Key Dataset Variables

| Variable | Role |
|---|---|
| `Average_Ratings` | Dependent variable (target) |
| `Number_of_Riders` | Independent variable |
| `Expected_Ride_Duration` | Independent variable (proxy for Trip_Distance) |
| `Customer_Loyalty_Status` | Categorical: Gold / Silver / Regular |
| `Vehicle_Type` | Categorical: Economy / Premium |
| `Historical_Cost_of_Ride` | Not used in main models |

## Environment & Data Path

Notebooks were developed in Google Colab. The CSV is loaded from Google Drive:
```python
df = pd.read_csv("/content/drive/MyDrive/ESPECIALIZACIÓN/MÉTODOS ESTADISTICOS/Anexo 1 - Base de Datos dynamic_pricing.csv")
```
When running locally, update this path to `Data/Dataset_dynamic_pricing.csv`.

## Core Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
```

## Key Findings Summary

- R² of the OLS model is ~0.005 — the selected predictors have very little explanatory power over `Average_Ratings`
- Only `Customer_Loyalty_Status_Silver` is statistically significant (p ≈ 0.034)
- No significant differences found between loyalty/vehicle groups in duration or ratings
- Breusch-Pagan detected marginal heteroscedasticity (p ≈ 0.043) in the original OLS model
- Log transformation resolves heteroscedasticity (BP p ≈ 0.066) but does not improve R²
- Bootstrap CI for `Customer_Loyalty_Status_Silver`: [0.004, 0.137] — excludes zero
