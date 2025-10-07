# ==== GPU / Entorno: Diagnóstico rápido ====
import sys, os, platform, importlib, json, shutil, subprocess
print("==== System ====")
print(f"Python       : {sys.version.split()[0]}  | {platform.system()} {platform.release()} ({platform.machine()})")
print(f"Process/BLAS : NUM_THREADS={os.environ.get('OMP_NUM_THREADS','')}  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS','')}")

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10)
        print(out.strip())
    except Exception as e:
        print(f"[cmd failed] {cmd} -> {e}")

print("\n==== NVIDIA Drivers / CUDA Toolkit (sistema) ====")
if shutil.which("nvidia-smi"): 
    run("nvidia-smi -L")
    run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
else:
    print("nvidia-smi no encontrado (posible entorno sin drivers/CUDA del sistema).")

if shutil.which("nvcc"):
    print("\nCUDA toolkit encontrado:")
    run("nvcc --version")
else:
    print("\nCUDA toolkit (nvcc) no encontrado (no indispensable si usas ruedas precompiladas).")

print("\n==== PyTorch ====")
try:
    import torch
    print(f"torch        : {torch.__version__}")
    print(f"CUDA (build) : {torch.version.cuda}")
    print(f"cuDNN (build): {getattr(torch.backends.cudnn, 'version', lambda: None)()}")
    print(f"CUDA avail?  : {torch.cuda.is_available()}")
    print(f"cuDNN avail? : {torch.backends.cudnn.enabled}")
    print(f"GPUs visible : {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            print(f"  - GPU[{i}] = {torch.cuda.get_device_name(i)} | CC {cap[0]}.{cap[1]}")
        # smoke test en GPU (opcional, rápido)
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = torch.mm(x, x)
            torch.cuda.synchronize()
            print("  PyTorch CUDA test: OK (matmul)")
        except Exception as e:
            print(f"  PyTorch CUDA test: FAIL -> {e}")
    else:
        print("  No hay GPU disponible para PyTorch.")
except Exception as e:
    print(f"PyTorch not importable -> {e}")

print("\n==== NumPy / Pandas / scikit-learn ====")
try:
    import numpy as np; print("numpy        :", np.__version__)
except Exception as e:
    print("numpy        : not importable ->", e)
try:
    import pandas as pd; print("pandas       :", pd.__version__)
except Exception as e:
    print("pandas       : not importable ->", e)
try:
    import sklearn; print("scikit-learn :", sklearn.__version__, "(CPU; no usa GPU nativamente)")
except Exception as e:
    print("scikit-learn : not importable ->", e)

print("\n==== Optuna ====")
try:
    import optuna; print("optuna       :", optuna.__version__)
except Exception as e:
    print("optuna       : not importable ->", e)

print("\n==== Librerías con soporte GPU (si disponibles) ====")
# XGBoost
try:
    import xgboost as xgb
    print("xgboost      :", xgb.__version__)
    # Chequeo simple de build; el soporte real se valida al usar tree_method='gpu_hist'
    cfg = getattr(xgb, 'build_info', lambda: {})()
    if isinstance(cfg, dict):
        print("  xgboost build_info:", json.dumps(cfg, indent=2)[:300], "...")
    else:
        print("  xgboost build_info no disponible; prueba con tree_method='gpu_hist' al entrenar.")
except Exception as e:
    print("xgboost      : not importable ->", e)

# LightGBM
try:
    import lightgbm as lgb
    print("lightgbm     :", lgb.__version__)
    print("  Nota: El soporte GPU depende de compilación con OpenCL/CUDA; se valida al entrenar con device='gpu'.")
except Exception as e:
    print("lightgbm     : not importable ->", e)

# CatBoost
try:
    import catboost as cb
    print("catboost     :", cb.__version__)
    print("  Nota: Usa GPU con task_type='GPU'. Se valida al entrenar.")
except Exception as e:
    print("catboost     : not importable ->", e)

# RAPIDS (cuDF/cuML)
try:
    import cudf, cuml
    print("cudf         :", cudf.__version__)
    print("cuml         :", cuml.__version__)
except Exception as e:
    print("RAPIDS (cudf/cuml) : not importable ->", e)

print("\n==== Resumen corto ====")
def _yes(b): return "YES" if b else "NO"
try:
    import torch
    print(f"PyTorch CUDA usable : {_yes(torch.cuda.is_available())} | GPUs: {torch.cuda.device_count()}")
except:
    print("PyTorch CUDA usable : NO (torch no importable)")
print("Listo.")
