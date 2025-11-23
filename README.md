# ☀️ Short-Term Solar Irradiance Forecasting in Colombia

This repository contains the codebase supporting my undergraduate thesis on short-term (10–60 min) GHI forecasting for a reference PV site in El Paso, Cesar (Colombia).  
The project compares tabular, recurrent and satellite-based deep learning models, with a special focus on tropical cloud dynamics and operational forecasting needs.

---

## Overview

Colombia’s rapid growth in solar PV capacity has increased the sensitivity of the grid to fast irradiance fluctuations. Forecasting errors directly affect market performance under local regulation (CREG 060/2019), making short-term GHI prediction a high-impact operational task.

This project evaluates two main families of forecasting models:

- **Tabular models:** Linear Regression, Random Forest, and RNN variants (LSTM, GRU, Dilated RNN, Clockwork RNN) trained on engineered features from on-site measurements.
- **Satellite-based models:** GOES-16 DSRF and MCMIPF products integrated through ConvLSTM architectures.
- **Hybrid models:** Fusion of tabular and satellite inputs for improved performance under variable cloud conditions.

An AWS-based ingestion pipeline was implemented to automate multi-year retrieval, cropping and storage of GOES patches using Docker-based Lambda functions and S3 buckets.

---

## Main Findings (summary)

- A tuned **Random Forest** is the strongest baseline, achieving ~50–55 W/m² RMSE and R² ≈ 0.97.
- Pure ConvLSTM models capture cloud dynamics but underperform compared to the best tabular approaches.
- **Hybrid (Tabular + DSRF)** models reduce errors relative to pure satellite models and approach RNN-based tabular models.
- Satellite information is most useful in **intermediate GHI regimes**, where cloud variability is highest.

---

## Repository Structure

```bash
solar-forecasting-colombia/
│
├── data/             # Raw, clean and GOES_v2 structures (empty for confidentiality)
├── notebooks/        # Exploration, modeling and diagnostics notebooks
├── src/              # Modular pipeline: preprocessing, features, models, utils
├── outputs/          # Generated figures and artifacts (empty / lightweight only)
├── tests/            # Basic tests for pipeline components
├── environment.yml   # Conda environment (preferred)
├── requirements.txt  # Minimal pip dependencies
└── README.md
```
### Data Directory

This repository does **not** include real datasets for confidentiality and size reasons.

The folder structure mirrors the structure used during development so that the
pipeline remains fully reproducible if data is provided.
