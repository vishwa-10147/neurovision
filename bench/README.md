# Benchmarks with real vision models

Scripts behind [`docs/vision-benchmarks.md`](../docs/vision-benchmarks.md).

| Script | Runs in | What it does |
|---|---|---|
| `vision_models.py` | `bench/.venv-gpu` (CUDA PyTorch) | Pretrained torchvision models: params, GMACs, predictions on real photos, GPU/CPU latency, fp16 throughput, CPU/GPU load, peak memory; exports ONNX to `exports/` |
| `check_onnx.py` | `backend/.venv` (ONNX Runtime) | Compares each export with PyTorch and times it on the CPU |
| `train_live.py` | `bench/.venv-gpu` | Fine-tunes a model on scikit-learn's digits on the GPU, streaming metrics, health and system use to Live Runs |

Set up the GPU environment (kept out of git, as are `weights/` and `exports/`):

``` bash
python -m venv bench/.venv-gpu
bench/.venv-gpu/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
bench/.venv-gpu/Scripts/pip install psutil nvidia-ml-py onnx scikit-learn
```

`vision_models.py` downloads about 800 MB of pretrained weights once (to
`bench/weights/`). Importing exports over 64 MB needs the backend started
with a higher `NEUROVISION_MAX_UPLOAD_BYTES`, e.g. 536870912.

`train_live.py` options: `--model`, `--epochs`, `--batch`, `--lr`,
`--log-every`, `--bn-momentum`, `--no-health`, `--url` (default
`http://127.0.0.1:8001`).
