# Demos: real models training live in NeuroVision

Each demo trains a real PyTorch model and streams it to **Live runs**,
where you see, while it trains:

-   the **network's graph** in 2D and 3D, colored by each layer's
    activation size, gradient norm or dead units, updated every step;
-   loss and accuracy (training and validation) as charts;
-   post-activation values of chosen layers for one fixed example;
-   **health alerts** (exploding/vanishing gradients, dead or saturated
    units, loss spikes, overfitting, …);
-   CPU, RAM and GPU load, memory, temperature and power, step time,
    throughput and **ETA**;
-   **controls**: pause, resume, stop and change the learning rate.

| Demo | Model | Data | Result (RTX 4070 Laptop GPU) |
|---|---|---|---|
| `train_spiral.py` | MLP 2 → 32 → 16 → 3 (ReLU, tanh) | 3-arm spiral, generated | 97.8 % validation accuracy, 150 epochs |
| `train_digits_cnn.py` | CNN: 2 conv blocks + classifier | scikit-learn digits (1,797 images) | 98.6 % in 3.3 s (30 epochs) |
| `train_attention.py` | Self-attention (4 heads) + feed-forward | "is the first token repeated?", generated | 99.2 % in 19 s (40 epochs) |
| `train_autoencoder.py` | 64 → 32 → **8** → 32 → 64 | scikit-learn digits | reconstruction MSE 0.023 in 5.9 s |
| `../../bench/train_live.py` | ResNet, MobileNet, EfficientNet, ConvNeXt, ViT (pretrained) | digits at 224×224 | 98.6–99.7 %, see [vision benchmarks](../../docs/vision-benchmarks.md) |

All run on the CPU too (`--device cpu`), more slowly.

## Run one

Start the backend and frontend (see the main README), open
**Live runs**, then:

``` bash
pip install torch scikit-learn psutil nvidia-ml-py   # psutil / nvidia-ml-py: CPU and GPU figures
python sdk/examples/train_digits_cnn.py --url http://127.0.0.1:8000
```

Options for the three new demos: `--epochs`, `--lr`, `--batch`,
`--delay` (seconds per epoch, to watch slowly), `--device auto|cpu|cuda`,
`--no-graph`.

## What each one shows

-   **Digits CNN:** in *Activation size* the three ReLU outputs are
    shaded, the classifier's ReLU strongest; in *Gradient norm* the last
    linear layer carries the largest gradients; selecting the first ReLU
    showed 6 % of its channels dead on the probe batch.
-   **Attention:** the graph shows Q, K and V projections in parallel,
    the scores' softmax and the feed-forward block. The attention is
    written out step by step because PyTorch's `nn.MultiheadAttention`
    uses a fused kernel in eval mode that cannot be exported to ONNX.
    Validation accuracy stays near chance (50–55 %) for about 9 epochs,
    then climbs from 59 % to 97 % between epochs 10 and 16, once the
    model learns to attend to the first token.
-   **Autoencoder:** the 8-value tanh code saturates (≈98 % of outputs
    at ±1), and NeuroVision raises a *saturated units* alert: a real
    finding about this bottleneck, worth comparing with a linear code.
-   **Spiral:** run with `--lr 0.5` to watch the health alerts fire.

## Use it in your own script

``` python
import neurovision_client as nv

probe = nv.ActivationProbe.by_name(model, ["features.1", "classifier.2"], x_val[:1])
monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])
with nv.start_run(
    "my model",
    total_epochs=EPOCHS,
    url=URL,
    model=model,
    example_input=x_val[:1],  # links the graph
    control=True,
    optimizer=optimizer,
) as run:
    for epoch in range(1, EPOCHS + 1):
        for xb, yb in loader:
            if not run.poll():  # pause / stop from the dashboard
                break
            ...  # forward, backward, optimizer.step()
        run.log(
            epoch=epoch,
            metrics={"loss": loss, "val_loss": val_loss},
            activations=probe.capture(),
            health=monitor.collect(),
            samples=len(train_set),
        )
```

`demo_common.py` is this loop, shared by the demos.
