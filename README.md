# ☀️ Comparing Recurrent Neural Architectures for Solar Forecasting in Tropical Regions

This project aims to compare multiple recurrent neural network architectures — including LSTM, GRU, Dilated RNN, and Clockwork RNN — to evaluate their effectiveness in short-term solar irradiance forecasting in Colombia's highly variable tropical climate.

⚡ **Goal**: Reduce forecast uncertainty and improve market performance for photovoltaic (PV) plants by identifying the most accurate and reliable RNN-based model architectures.

---

## 🌞 Motivation

Colombia is advancing its energy transition by integrating solar PV into the national grid. However, regulatory mechanisms such as **CREG 060 of 2019** impose penalties for deviations between forecasted and actual energy delivery in the short-term electricity market. This creates a pressing need for accurate, context-adapted irradiance forecasting models.

Traditional models like **GFS** struggle in tropical contexts due to their coarse spatial resolution and inability to capture rapid atmospheric changes. This project proposes a systematic comparison of recurrent deep learning models to support **informed operational planning**, reduce uncertainty, and optimize investment decisions in renewable energy.

---

## 🧠 Core Ideas

- **Architectural comparison**: LSTM, GRU, Dilated RNN, and Clockwork RNN evaluated side-by-side.
- **Forecast horizons**: Real-time, intra-day (4h), and day-ahead (24h) prediction windows.
- **Feature engineering**: Includes cyclic time variables, statistical windows, and variable interactions.
- **Application-oriented**: Focused on practical implementation for PV plants under Colombian regulation.
- **Replicability**: Designed to be adaptable to multiple solar generation sites.

---

## 🗂 Project Structure

```bash
solar-forecasting-colombia/
│
├── data/             # Raw, processed and external data sources
├── notebooks/        # Jupyter notebooks for exploration and experimentation
├── src/              # Python modules: preprocessing, features, models, utils
├── outputs/          # Generated plots, results, and model checkpoints
├── tests/            # Optional test scripts for functions and modules
├── requirements.txt  # Python dependencies
├── .gitignore        # Files excluded from version control
└── README.md         # This file
