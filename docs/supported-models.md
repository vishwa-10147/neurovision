# Which models NeuroVision supports

NeuroVision works with models in five ways. Each has its own scope.

| Where | What you can bring | What you get |
|---|---|---|
| **Architecture Studio + Training Lab** | Dense (fully connected) networks you design in the app: 1–16 inputs, up to 6 hidden layers of 1–32 units, 1–10 outputs, ReLU / tanh / sigmoid | Training in the browser's backend on moons, circles or spiral data; every unit, weight, activation and gradient in 2D and 3D |
| **Model Explorer** | Any trained model exported to **ONNX**: from PyTorch, TensorFlow/Keras (tf2onnx) or scikit-learn (skl2onnx) | Graph in 2D and 3D, weights, running the model on your input or a photo, every operator's output, feature maps, attention heads, conv filters |
| **Classical ML** | Nine scikit-learn algorithms: logistic regression, linear and RBF SVMs, k-NN, decision tree, random forest, gradient boosting, k-Means, PCA ([guide](classical-ml.md)) | Decision regions animated over epochs, trees, k or iterations; the fitted tree, support vectors, centroids, principal components |
| **Neural Lab** | Built-in PyTorch models you configure: MLP, CNN, LSTM/GRU/RNN, transformer, autoencoder, VAE, GAN, GNN, diffusion, REINFORCE ([guide](neural-lab.md)) | Trained live in Live Runs with graph, visuals, health and controls |
| **Live Runs** | Any **PyTorch** training script, on CPU or GPU, via the one-file SDK | Live metrics, the network's graph colored as it trains, custom visuals, health alerts, CPU/GPU/ETA, pause/stop/learning-rate control |

## Model families, verified

Every row was run in this repository; the numbers come from
[vision benchmarks](vision-benchmarks.md) and the [demos](../sdk/examples/README.md).

| Family | Example | Model Explorer (ONNX) | Live Runs + live graph |
|---|---|---|---|
| MLP | spiral classifier, autoencoder | ✓ | ✓ demos `train_spiral.py`, `train_autoencoder.py` |
| CNN | digits CNN, ResNet-18/50, MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny | ✓ feature maps, filters | ✓ demo `train_digits_cnn.py`; `bench/train_live.py` on GPU |
| Transformer / attention | self-attention classifier, ViT-B/16 (1,016 operators) | ✓ attention heads (captured on request above 200 operators) | ✓ demo `train_attention.py` (attention written out, see below) |
| 3D CNN (video, volumes) | R3D-18 | ✓ 5D maps, one per frame | ✓ (PyTorch script; same SDK) |
| Recurrent (LSTM, GRU, RNN) | forecaster | ✓ (exports as LSTM/GRU/RNN operators) | ✓ demo `train_lstm.py`, Neural Lab |
| Generative (VAE, GAN, diffusion) | digits VAE, digits GAN, 2D DDPM | ✓ | ✓ demos with latent space, samples |
| Graph neural network | GCN on the karate club | ✓ (dense adjacency as MatMul) | ✓ demo `train_gnn.py` |
| Reinforcement learning | REINFORCE on CartPole | ✓ policy network | ✓ demo `train_cartpole.py` |
| Classical ML | trees, forests, boosting, linear models, SVMs, k-NN (skl2onnx) | ✓ graph, running, **tree view** | Classical ML Lab (in-app fitting) |

Sizes: graphs up to 5,000 operators; files up to 64 MB by default
(`NEUROVISION_MAX_UPLOAD_BYTES`; 512 MB covered ViT-B/16's 331 MB). The
3D view draws up to 1,500 nodes; larger graphs use 2D.

## Linking a live run to its graph

`nv.start_run(..., model=model, example_input=x[:1])` exports the model
with PyTorch's TorchScript ONNX exporter, imports it, and links it to the
run. If the export or the upload fails, the run still starts, without a
graph, and the reason is logged. What can stop the graph:

| Case | What to do |
|---|---|
| `nn.MultiheadAttention` in eval mode (PyTorch's fused kernel `_native_multi_head_attention` has no ONNX export) | Write attention out (Q/K/V projections, scores, softmax), as `train_attention.py` does; the graph then shows each step |
| Data-dependent Python control flow | Export traces one path; the graph shows that path |
| Custom autograd functions or ops without an ONNX symbolic | No graph; metrics, health and system figures still stream |
| Export larger than the server's upload limit | Raise `NEUROVISION_MAX_UPLOAD_BYTES` on the backend |
| Shapes computed from `x.shape` inside `forward` | Works, but adds shape-arithmetic operators; fixed sizes (`reshape(-1, L, H, D)`) give a cleaner graph |

Layers are placed on the graph **by module name**: PyTorch names each
ONNX node after its module (`/features/features.1/Relu`), and the SDK
reports statistics by module name (`features.1`). Use
`ActivationProbe.by_name(model, ["features.1", ...], x)`; `HealthMonitor`
already uses module names. Operators whose module reported nothing stay
uncolored, and names that match no operator are listed under the graph.

## Other frameworks

-   **TensorFlow / Keras, JAX:** export to ONNX for the Model Explorer.
    For Live Runs, `run.log(epoch=..., metrics={...})` accepts plain
    Python floats, so loss, accuracy and system/ETA figures stream from
    any framework; the graph link, `ActivationProbe` and `HealthMonitor`
    need PyTorch.
-   **Large language models:** importable as ONNX if under the size and
    5,000-operator limits; there is no tokenizer, so attention axes are
    positions, not words.
