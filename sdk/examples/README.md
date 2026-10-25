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
| `train_lstm.py` | LSTM / GRU / RNN (`--cell`), 30-step window | generated seasonal series | validation MSE 0.021 (standardized) |
| `train_gan.py` | generator + discriminator (MLPs) | scikit-learn digits | recognizable 8×8 digits after 150 epochs |
| `train_vae.py` | VAE, 2D latent space | scikit-learn digits | digit classes cluster without labels; loss 24.4 |
| `train_gnn.py` | 2-layer graph convolutional network | Zachary's karate club (real, via networkx) | 94–97 % of 34 members from 2 labels |
| `train_diffusion.py` | DDPM denoiser, 100 noise levels | generated spiral | samples form the spiral after ~300 epochs |
| `train_cartpole.py` | REINFORCE policy 4 → 64 → 2 | simulated cart-pole (no gym needed) | 500-step episodes, the maximum |
| `train_lightning.py` | PyTorch Lightning MLP | generated binary classes | framework callback, loss/validation/health |
| `train_hf_trainer.py` | Hugging Face tiny BERT classifier | generated token rows, no downloads | framework callback, validation metrics |
| `train_keras.py` | Keras Dense 3 → 8 → 2 | generated binary classes | framework callback, validation metrics |
| `../../bench/train_live.py` | ResNet, MobileNet, EfficientNet, ConvNeXt, ViT (pretrained) | digits at 224×224 | 98.6–99.7 %, see [vision benchmarks](../../docs/vision-benchmarks.md) |

All run on the CPU too (`--device cpu`), more slowly.

## Run one

Start the backend and frontend (see the main README), open
**Live runs**, then:

``` bash
pip install ./sdk torch scikit-learn psutil nvidia-ml-py   # psutil / nvidia-ml-py: CPU and GPU figures
python sdk/examples/train_digits_cnn.py --url http://127.0.0.1:8000
```

For the framework examples, install the matching optional extra:

``` bash
pip install "./sdk[lightning]"       # train_lightning.py
pip install "./sdk[transformers]"    # train_hf_trainer.py
pip install "./sdk[keras]"           # train_keras.py
```

Every demo takes `--epochs`, `--lr`, `--delay` (seconds per epoch, to
watch slowly), `--name`, `--device auto|cpu|cuda` and `--no-graph`; some
take their own: `train_digits_cnn.py --blocks --channels --hidden`,
`train_lstm.py --cell lstm|gru|rnn --hidden --layers`,
`train_attention.py --dim --heads --blocks`, `train_autoencoder.py --code`.
The [Neural Lab](../../docs/neural-lab.md) page sets these from the
browser and starts the script for you.

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

-   **LSTM / GRU / RNN:** the forecast against held-out values (blue =
    actual, red = forecast) and the hidden state at every time step.
-   **GAN:** 16 digits generated from the same fixed noise each epoch;
    watch blobs turn into digits while the two losses balance.
-   **VAE:** every held-out digit's 2D code, colored by its true class:
    the classes separate although the model never sees labels.
-   **GNN:** each member's 2D embedding splits into the two factions.
-   **Diffusion:** fresh samples every 10 epochs; the loss barely moves
    while the samples go from a blob to a spiral.
-   **CartPole:** the pole angle over the latest episode; episodes grow
    from ~20 steps to the 500-step maximum.

## Custom visuals

Any script can attach scatter plots and heatmaps to a step:

``` python
run.log(
    epoch=epoch,
    metrics={"loss": loss},
    visuals={
        "latent space": nv.scatter(z[:, 0], z[:, 1], labels=y),  # ≤ 512 points
        "attention": nv.matrix(weights[0], scale="sequential"),  # ≤ 64 × 64
        "samples": nv.matrix(nv.image_grid(images, columns=4), scale="sequential"),
    },
)
```

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
