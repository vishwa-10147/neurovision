# neurovision-client

`neurovision-client` streams metrics, selected activations and training health
from a training script to NeuroVision's **Live Runs** view. The base package has
no required dependencies and works with a plain Python loop. Framework
callbacks are optional extras, so installing the client does not force a
project to install Lightning, Transformers or Keras.

## Install

From a checkout:

```bash
pip install ./sdk
```

With a framework extra from GitHub:

```bash
pip install "neurovision-client[lightning] @ git+https://github.com/vishwa-10147/neurovision#subdirectory=sdk"
```

Optional extras are `torch`, `system`, `lightning`, `transformers` and `keras`.
The package itself only declares the Python standard library as required.
Copying `neurovision_client.py` and `neurovision_callbacks.py` beside a script
continues to work for users who do not want a package install.

## Framework callbacks

Each framework adds one callback to its normal trainer:

```python
from neurovision_callbacks import LightningCallback

trainer = pl.Trainer(callbacks=[LightningCallback(url=URL)])
```

```python
from neurovision_callbacks import HuggingFaceCallback

trainer = Trainer(callbacks=[HuggingFaceCallback(url=URL)])
```

```python
from neurovision_callbacks import KerasCallback

model.fit(x, y, callbacks=[KerasCallback(url=URL)])
```

The callbacks lazily import their framework. Importing
`neurovision_callbacks` in a minimal environment therefore does not require
any of the three frameworks. See [`../docs/live-runs.md`](../docs/live-runs.md)
for controls, activation probes, checkpoint options, examples and known
limitations.

## Plain PyTorch loops

```python
import neurovision_client as nv

with nv.start_run("my experiment", total_epochs=10, url=URL) as run:
    for epoch in range(1, 11):
        # train one epoch here
        run.log(epoch=epoch, metrics={"loss": loss, "accuracy": accuracy})
```

The client is intentionally tolerant of a missing backend: it warns and lets
the training script continue locally.

## Verify a wheel offline

From `backend/` with the project's development virtual environment active:

```powershell
.venv/Scripts/python -m pip wheel --no-build-isolation --no-deps ../sdk -w <scratch>/dist
python -m venv <scratch>/clean
<scratch>/clean/Scripts/python -m pip install --no-index <scratch>/dist/neurovision_client-0.1.0-py3-none-any.whl
<scratch>/clean/Scripts/python -c "import neurovision_client, neurovision_callbacks; print(neurovision_client.__version__)"
```

The final command prints `0.1.0` without installing any framework. Framework
extras are intentionally separate from the base wheel.
