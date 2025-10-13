# %% [markdown]
# # 03 — Modeling Tabular (Keras/TF)

# %% [markdown]
# ## Baselines + RNNs (LSTM, GRU, Dilated, Clockwork).
# 
# **Optuna (JournalStorage + lock)**

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import os, json, math, time, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, backend as K

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.integration import TFKerasPruningCallback
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.storages.journal._file import JournalFileOpenLock

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## Config

# %%
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
print("TF GPUs:", gpus)

# %%
DATA_CLEAN = Path("../data/clean/base_dataset.csv")
OUT_DIR = Path("../outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = OUT_DIR / "artifacts_keras"; ART_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
TARGET_COL = "GHI"
FREQ = "10T"
DEFAULT_INPUT_STEPS   = 36   # 6h pasado
DEFAULT_HORIZON_STEPS = 6    # 1h adelante

PATIENCE = 8

# %% [markdown]
# ### Mode

# %%
MODE = "AIA"  # "LAPTOP" o "AIA"

if MODE == "LAPTOP":
    N_TRIALS_RF   = 30
    N_TRIALS_LSTM = 40
    N_TRIALS_GRU  = 40
    N_TRIALS_DIL  = 35
    N_TRIALS_CW   = 35
    MAX_EPOCHS    = 60
elif MODE == "AIA":
    N_TRIALS_RF   = 120
    N_TRIALS_LSTM = 120
    N_TRIALS_GRU  = 120
    N_TRIALS_DIL  = 120
    N_TRIALS_CW   = 120
    MAX_EPOCHS    = 90

# %%
PRUNER = MedianPruner(n_warmup_steps=5)

# %% [markdown]
# ### Storage

# %%
JOURNAL_PATH = (OUT_DIR / "optuna_tabular_keras.journal").resolve()
LOCK = JournalFileOpenLock(str(JOURNAL_PATH) + ".lock")
STORAGE = JournalStorage(JournalFileStorage(str(JOURNAL_PATH), lock_obj=LOCK))

# %%
def prepare_journal_storage(study_name: str) -> JournalStorage:
    log_path = (OUT_DIR / f"{study_name}.log").resolve()
    lock     = JournalFileOpenLock(str(log_path) + ".lock")
    return JournalStorage(JournalFileStorage(str(log_path), lock_obj=lock))

# %% [markdown]
# ## Data

# %%
df = pd.read_csv(DATA_CLEAN, parse_dates=[0], index_col=0).sort_index()
df.index.name = "time"

# %%
base_feats = [
    'Presion','TempAmb','WindSpeed','WindDirection',
    'hour_sin','hour_cos','DoY Sin','DoY Cos',
    'solar_zenith','solar_azimuth','solar_elevation',
    'TempAmb_roll1h_mean','TempAmb_roll6h_mean',
    'Presion_roll1h_mean','Presion_roll6h_mean',
    'WindSpeed_roll1h_mean','WindSpeed_roll6h_mean',
    'temp_pressure_ratio','wind_temp_interaction'
]
ghi_lags  = [c for c in ['GHI_lag1','GHI_lag3','GHI_lag6','GHI_lag12','GHI_lag36'] if c in df.columns]
ghi_rolls = [c for c in ['GHI_roll1h_mean','GHI_roll3h_mean','GHI_roll6h_mean','GHI_roll1h_max'] if c in df.columns]
feat_cols = [c for c in base_feats if c in df.columns] + ghi_lags + ghi_rolls
print(f"Total features used: {len(feat_cols)}")
print(feat_cols)

# %%
assert TARGET_COL in df.columns, f"TARGET_COL='{TARGET_COL}' no existe en el dataset"
n = len(df); i_tr = int(0.7*n); i_va = int(0.85*n)
df_train, df_val, df_test = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]

X_scaler = StandardScaler(); y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(df_train[feat_cols].values)
X_val   = X_scaler.transform(df_val[feat_cols].values)
X_test  = X_scaler.transform(df_test[feat_cols].values)

y_train = y_scaler.fit_transform(df_train[[TARGET_COL]].values).ravel()
y_val   = y_scaler.transform(df_val[[TARGET_COL]].values).ravel()
y_test  = y_scaler.transform(df_test[[TARGET_COL]].values).ravel()

# %%
print("NaNs antes de imputar:",
      np.isnan(X_train).sum(), np.isnan(X_val).sum(), np.isnan(X_test).sum())

# %%
imp = SimpleImputer(strategy="median")
X_train = imp.fit_transform(X_train)
X_val   = imp.transform(X_val)
X_test  = imp.transform(X_test)

for name, arr in [("X_train",X_train),("X_val",X_val),("X_test",X_test),
                  ("y_train",y_train),("y_val",y_val),("y_test",y_test)]:
    assert np.isfinite(arr).all(), f"{name} tiene NaN/Inf"

# %% [markdown]
# ## Helpers

# %%
def metrics_from_scaled(pred_scaled, true_scaled, y_scaler):
    p = y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
    t = y_scaler.inverse_transform(true_scaled.reshape(-1,1)).ravel()
    mae = mean_absolute_error(t, p)
    rmse = float(np.sqrt(mean_squared_error(t, p)))
    mape = float(np.mean(np.abs((t + 1e-6) - p) / (np.abs(t) + 1e-6)) * 100)
    smape = float(100 * np.mean(2*np.abs(p - t) / (np.abs(t) + np.abs(p) + 1e-6)))
    r2 = float(r2_score(t, p))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2}, (t, p)

def persistence_baseline(y_scaled, horizon):
    y_hat = np.roll(y_scaled, horizon)
    y_hat[:horizon] = y_scaled[horizon]  
    return y_hat

def _rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))

# %% [markdown]
# ### sequences

# %%
def build_seq_arrays(X_2d, y_1d, L, horizon):
    """
    X_2d: (N, F), y_1d: (N,), L: window len (input_steps), horizon: steps ahead
    Devuelve X_seq (N', L, F), y_seq (N',)
    """
    N, F = X_2d.shape
    outX, outy = [], []
    last = N - L - horizon + 1
    if last <= 0:
        return np.zeros((0, L, F), dtype="float32"), np.zeros((0,), dtype="float32")
    for i in range(last):
        block = X_2d[i:i+L]
        if np.isnan(block).any():
            continue
        outX.append(block)
        outy.append(y_1d[i + L + horizon - 1])
    return np.asarray(outX, dtype="float32"), np.asarray(outy, dtype="float32")

# %% [markdown]
# ## Baselines

# %%
lin = LinearRegression().fit(X_train, y_train)
lin_metrics, (y_true_lin, y_pred_lin) = metrics_from_scaled(lin.predict(X_test), y_test, y_scaler)

rf0 = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
rf0_metrics, (y_true_rf0, y_pred_rf0) = metrics_from_scaled(rf0.predict(X_test), y_test, y_scaler)

y_pers_test = persistence_baseline(y_test, DEFAULT_HORIZON_STEPS)
pers_metrics, (y_true_pers, y_pred_pers) = metrics_from_scaled(y_pers_test, y_test, y_scaler)

print("Persistence:", pers_metrics)
print("Linear     :", lin_metrics)
print("RF baseline:", rf0_metrics)

# %% [markdown]
# ## Models - Keras

# %%
def build_lstm(L, n_feat, units=64, layers_n=1, dropout=0.0, bidir=False):
    inp = layers.Input(shape=(L, n_feat))
    x = inp
    for i in range(layers_n-1):
        cell = layers.LSTM(units, return_sequences=True, dropout=dropout)
        if bidir: cell = layers.Bidirectional(cell)
        x = cell(x)
    # capa final
    cell = layers.LSTM(units, dropout=dropout)
    if bidir: cell = layers.Bidirectional(cell)
    x = cell(x)
    out = layers.Dense(1, dtype="float32")(x)
    return models.Model(inp, out)

# %%
def build_gru(L, n_feat, units=64, layers_n=1, dropout=0.0, bidir=False):
    inp = layers.Input(shape=(L, n_feat))
    x = inp
    for i in range(layers_n-1):
        cell = layers.GRU(units, return_sequences=True, dropout=dropout)
        if bidir: cell = layers.Bidirectional(cell)
        x = cell(x)
    cell = layers.GRU(units, dropout=dropout)
    if bidir: cell = layers.Bidirectional(cell)
    x = cell(x)
    out = layers.Dense(1, dtype="float32")(x)
    return models.Model(inp, out)

# %%
def build_dilated_like(L, n_feat, units=64, dilation=2, dropout=0.0):
    assert dilation >= 1
    inp = layers.Input(shape=(L, n_feat))
    # Submuestreo por "dilation"
    x = layers.Lambda(lambda t: t[:, ::dilation, :])(inp)
    x = layers.LSTM(units, dropout=dropout)(x)
    out = layers.Dense(1, dtype="float32")(x)
    return models.Model(inp, out)

# %%
def build_clockwork(L, n_feat, hidden=60, modules=3, base_period=1, dropout=0.0):
    assert hidden % modules == 0
    h_per = hidden // modules
    periods = [base_period * (2**m) for m in range(modules)]
    inp = layers.Input(shape=(L, n_feat))
    h_list = []
    for p in periods:
        # Submuestrea según periodo p (t%p==0) ~ aproximación
        xt = layers.Lambda(lambda t, step=p: t[:, ::step, :])(inp)
        ht = layers.SimpleRNN(h_per, activation="tanh", dropout=dropout)(xt)
        h_list.append(ht)
    h = layers.Concatenate()(h_list) if len(h_list) > 1 else h_list[0]
    out = layers.Dense(1, dtype="float32")(h)
    return models.Model(inp, out)

# %% [markdown]
# ## Optuna

# %%
def make_seq_data_for_trial(steps, horizon):
    Xtr_seq, ytr_seq = build_seq_arrays(X_train, y_train, steps, horizon)
    Xva_seq, yva_seq = build_seq_arrays(X_val,   y_val,   steps, horizon)
    if min(len(Xtr_seq), len(Xva_seq)) == 0:
        raise optuna.TrialPruned()
    return Xtr_seq, ytr_seq, Xva_seq, yva_seq

# %%
def objective_rf(trial: optuna.Trial) -> float:
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 6, 28),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        n_jobs=-1, random_state=SEED
    )
    rf.fit(X_train, y_train)
    pred_val = rf.predict(X_val)
    p_o = y_scaler.inverse_transform(pred_val.reshape(-1,1)).ravel()
    t_o = y_scaler.inverse_transform(y_val.reshape(-1,1)).ravel()
    return float(np.sqrt(mean_squared_error(t_o, p_o)))

# %%
# LSTM/GRU generic
def objective_rnn(trial: optuna.Trial, kind="lstm") -> float:
    steps   = trial.suggest_categorical("input_steps",  [24, 36, 48, 60, 72])
    horizon = trial.suggest_categorical("horizon_steps",[3, 6, 12])
    units   = trial.suggest_int("hidden", 64, 256, step=32)
    layers_n= trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    bidir   = False
    lr      = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs      = trial.suggest_categorical("batch", [64, 128, 256, 512])
    eps     = trial.suggest_int("epochs", 40, MAX_EPOCHS)

    Xtr_seq, ytr_seq, Xva_seq, yva_seq = make_seq_data_for_trial(steps, horizon)
    n_feat = Xtr_seq.shape[2]

    if kind == "lstm":
        model = build_lstm(steps, n_feat, units=units, layers_n=layers_n, dropout=dropout, bidir=bidir)
    else:
        model = build_gru(steps, n_feat, units=units, layers_n=layers_n, dropout=dropout, bidir=bidir)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    tmp_dir = (ART_DIR / f"{kind}_t{trial.number:04d}"); tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss", save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    # Validación en espacio ORIGINAL
    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    p_o = y_scaler.inverse_transform(yhat.reshape(-1,1)).ravel()
    t_o = y_scaler.inverse_transform(yva_seq.reshape(-1,1)).ravel()
    val_rmse = float(np.sqrt(mean_squared_error(t_o, p_o)))

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("seq_len_used", steps)
    trial.set_user_attr("horizon_used", horizon)
    trial.set_user_attr("n_feat", n_feat)
    trial.set_user_attr("arch", kind.upper())
    return val_rmse

# %%
# Dilated-like
def objective_dilated(trial: optuna.Trial) -> float:
    steps   = trial.suggest_categorical("input_steps",  [24, 36, 48, 60, 72])
    horizon = trial.suggest_categorical("horizon_steps",[3, 6, 12])
    units   = trial.suggest_int("hidden", 64, 256, step=32)
    dilation= trial.suggest_categorical("dilation", [1, 2, 3, 4, 6])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr      = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs      = trial.suggest_categorical("batch", [64, 128, 256, 512])
    eps     = trial.suggest_int("epochs", 40, MAX_EPOCHS)

    Xtr_seq, ytr_seq, Xva_seq, yva_seq = make_seq_data_for_trial(steps, horizon)
    n_feat = Xtr_seq.shape[2]
    model = build_dilated_like(steps, n_feat, units=units, dilation=dilation, dropout=dropout)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    tmp_dir = (ART_DIR / f"dilated_t{trial.number:04d}"); tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss", save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    p_o = y_scaler.inverse_transform(yhat.reshape(-1,1)).ravel()
    t_o = y_scaler.inverse_transform(yva_seq.reshape(-1,1)).ravel()
    val_rmse = float(np.sqrt(mean_squared_error(t_o, p_o)))

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("seq_len_used", steps)
    trial.set_user_attr("horizon_used", horizon)
    trial.set_user_attr("n_feat", n_feat)
    trial.set_user_attr("arch", "DILATED")
    return val_rmse

# %%
# Clockwork
def objective_clockwork(trial: optuna.Trial) -> float:
    steps   = trial.suggest_categorical("input_steps",  [24, 36, 48, 60, 72])
    horizon = trial.suggest_categorical("horizon_steps",[3, 6, 12])
    hidden  = trial.suggest_int("hidden", 90, 300, step=30)
    modules = trial.suggest_categorical("modules", [3, 4, 5])
    if hidden % modules != 0:
        raise optuna.TrialPruned()
    base_p  = trial.suggest_categorical("base_period", [1, 2])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr      = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs      = trial.suggest_categorical("batch", [64, 128, 256, 512])
    eps     = trial.suggest_int("epochs", 40, MAX_EPOCHS)

    Xtr_seq, ytr_seq, Xva_seq, yva_seq = make_seq_data_for_trial(steps, horizon)
    n_feat = Xtr_seq.shape[2]
    model = build_clockwork(steps, n_feat, hidden=hidden, modules=modules, base_period=base_p, dropout=dropout)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    tmp_dir = (ART_DIR / f"clock_t{trial.number:04d}"); tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss", save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    p_o = y_scaler.inverse_transform(yhat.reshape(-1,1)).ravel()
    t_o = y_scaler.inverse_transform(yva_seq.reshape(-1,1)).ravel()
    val_rmse = float(np.sqrt(mean_squared_error(t_o, p_o)))

    # defensas extra (evita None/NaN/Inf)
    if not np.isfinite(val_rmse):
        raise optuna.TrialPruned()

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("seq_len_used", steps)
    trial.set_user_attr("horizon_used", horizon)
    trial.set_user_attr("n_feat", n_feat)
    trial.set_user_attr("arch", "CLOCKWORK")

    return val_rmse

# %% [markdown]
# ## Execution

# %%
# def run_study(name, obj_fn, n_trials):
#     print(f"→ Running {name} …")
#     storage = STORAGE  # o: prepare_journal_storage(name)
#     study = optuna.create_study(direction="minimize",
#                                 sampler=TPESampler(seed=SEED),
#                                 pruner=PRUNER,
#                                 study_name=name,
#                                 storage=storage,
#                                 load_if_exists=True)
#     study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=True)
#     print(f"{name} best:", study.best_trial.value, study.best_trial.params)
#     return study

def run_study(name, obj_fn, n_trials):
    print(f"→ Running {name} …")
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=PRUNER,
        study_name=name,
        storage=STORAGE,
        load_if_exists=True
    )
    study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=True)

    # robustez: si no hay trials completados, no intentes best_trial
    completes = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completes:
        print(f"{name} best:", study.best_trial.value, study.best_trial.params)
    else:
        print(f"{name}: no completed trials (all pruned/failed).")
    return study

# %%
study_rf   = run_study("RF_RMSE", objective_rf, N_TRIALS_RF)

# %%
study_lstm = run_study("LSTM_MSEval", lambda t: objective_rnn(t,"lstm"), N_TRIALS_LSTM)

# %%
study_gru  = run_study("GRU_MSEval", lambda t: objective_rnn(t,"gru"), N_TRIALS_GRU)

# %%
study_dil  = run_study("DilatedRNN_MSEval", objective_dilated, N_TRIALS_DIL)

# %%
study_cw   = run_study("ClockworkRNN_MSEval", objective_clockwork, N_TRIALS_CW)

# %% [markdown]
# ### Results

# %%
print("Best LSTM      :", study_lstm.best_trial.params)
print("Best GRU       :", study_gru.best_trial.params)
print("Best Dilated   :", study_dil.best_trial.params)
print("Best Clockwork :", study_cw.best_trial.params)

# %%
# RF (tabular)
best_rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **study_rf.best_trial.params)
best_rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
rf_opt_metrics, (y_true_rf_opt, y_pred_rf_opt) = metrics_from_scaled(best_rf.predict(X_test), y_test, y_scaler)

# %% [markdown]
# ## Retrain

# %% [markdown]
# ### Helpers

# %%
def rebuild_and_train(best_trial, arch):
    steps   = best_trial.user_attrs.get("seq_len_used") or best_trial.params.get("input_steps", DEFAULT_INPUT_STEPS)
    horizon = best_trial.user_attrs.get("horizon_used") or best_trial.params.get("horizon_steps", DEFAULT_HORIZON_STEPS)
    Xtr_seq, ytr_seq = build_seq_arrays(np.vstack([X_train, X_val]),
                                        np.concatenate([y_train, y_val]),
                                        steps, horizon)
    Xva_seq, yva_seq = build_seq_arrays(X_val, y_val, steps, horizon)  # val para early stopping consistente
    n_feat = Xtr_seq.shape[2]

    p = best_trial.params
    lr = p.get("lr", 1e-3)
    bs = p.get("batch", 128)
    eps = min(p.get("epochs", MAX_EPOCHS), MAX_EPOCHS)

    if arch == "LSTM":
        model = build_lstm(steps, n_feat, units=p.get("hidden",64),
                           layers_n=p.get("num_layers",1),
                           dropout=p.get("dropout",0.0), bidir=False)
    elif arch == "GRU":
        model = build_gru(steps, n_feat, units=p.get("hidden",64),
                          layers_n=p.get("num_layers",1),
                          dropout=p.get("dropout",0.0), bidir=False)
    elif arch == "DILATED":
        model = build_dilated_like(steps, n_feat, units=p.get("hidden",64),
                                   dilation=p.get("dilation",2),
                                   dropout=p.get("dropout",0.0))
    elif arch == "CLOCKWORK":
        model = build_clockwork(steps, n_feat, hidden=p.get("hidden",120),
                                modules=p.get("modules",3),
                                base_period=p.get("base_period",1),
                                dropout=p.get("dropout",0.0))
    else:
        raise ValueError(arch)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    ckpt = (ART_DIR / f"best_{arch.lower()}.weights.h5").resolve()
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(ckpt), monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    # Eval test
    Xte_seq, yte_seq = build_seq_arrays(X_test, y_test, steps, horizon)
    yhat = model.predict(Xte_seq, verbose=0).squeeze()

    # A métricas originales
    p_o = y_scaler.inverse_transform(yhat.reshape(-1,1)).ravel()
    t_o = y_scaler.inverse_transform(yte_seq.reshape(-1,1)).ravel()
    mae  = mean_absolute_error(t_o, p_o)
    rmse = _rmse(t_o, p_o)
    mape = float(np.mean(np.abs((t_o + 1e-6) - p_o) / (np.abs(t_o) + 1e-6)) * 100)
    smape= float(100*np.mean(2*np.abs(p_o - t_o)/(np.abs(t_o)+np.abs(p_o)+1e-6)))
    r2   = r2_score(t_o, p_o)
    return {"MAE":mae,"RMSE":rmse,"MAPE":mape,"sMAPE":smape,"R2":r2}, (t_o, p_o)

# %%
lstm_metrics, (yt_lstm, yp_lstm) = rebuild_and_train(study_lstm.best_trial, "LSTM")
gru_metrics,  (yt_gru,  yp_gru)  = rebuild_and_train(study_gru.best_trial,  "GRU")
dil_metrics,  (yt_dil,  yp_dil)  = rebuild_and_train(study_dil.best_trial,  "DILATED")
cw_metrics,   (yt_cw,   yp_cw)   = rebuild_and_train(study_cw.best_trial,   "CLOCKWORK")

# %% [markdown]
# ## Results

# %%
results = {
    "Persistence": pers_metrics,
    "LinearRegression": lin_metrics,
    "RandomForest_baseline": rf0_metrics,
    "RandomForest_Optuna": rf_opt_metrics,
    "LSTM_Optuna": lstm_metrics,
    "GRU_Optuna":  gru_metrics,
    "DilatedRNN_Optuna": dil_metrics,
    "ClockworkRNN_Optuna": cw_metrics,
}
res_df = pd.DataFrame(results).T.sort_values("RMSE")
display(res_df.round(3))

with open(ART_DIR/"tabular_results_optuna_keras.json","w") as f:
    json.dump({k:{m:float(vv) for m,vv in v.items()} for k,v in results.items()}, f, indent=2)
print("Saved:", ART_DIR/"tabular_results_optuna_keras.json")

# %% [markdown]
# ## Plots

# %%
def plot_sample(y_true, y_pred, title, n=1000, fname=None):
    n = min(n, len(y_true))
    plt.figure(figsize=(11,3.8))
    plt.plot(y_true[:n], label="Real", lw=1.5)
    plt.plot(y_pred[:n], label="Pred", lw=1.2, alpha=0.9)
    plt.title(title); plt.xlabel("Time steps (10-min)"); plt.ylabel("GHI (W/m²)")
    plt.legend(frameon=False); plt.tight_layout()
    if fname: plt.savefig(fname, dpi=140)
    plt.show()

plot_sample(y_true_rf_opt, y_pred_rf_opt, "RandomForest Optuna — Test (sample)",
            fname=FIG_DIR / "pred_rf_opt_sample.png")

for name, (yt, yp) in {
    "LSTM": (yt_lstm, yp_lstm),
    "GRU":  (yt_gru,  yp_gru),
    "Dilated": (yt_dil, yp_dil),
    "Clockwork": (yt_cw, yp_cw),
}.items():
    plot_sample(yt, yp, f"{name} — Test (sample)",
                fname=FIG_DIR / f"pred_{name.lower()}_sample.png")


