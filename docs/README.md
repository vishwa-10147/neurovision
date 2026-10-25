# NeuroVision documentation

Start with the project [README](../README.md) for setup. The documents below go deeper.

For an overview of the whole system with its design rationale and measured evaluation, read the technical paper, [`paper/NeuroVision_Paper.pdf`](../paper/NeuroVision_Paper.pdf).

## Guides

| Guide | Read it when you want to… |
|---|---|
| [Inspecting and running your own models (ONNX)](importing-onnx.md) | export a PyTorch model to ONNX, import it, inspect its graph and weights, and run it to see feature maps, filters and attention heads |
| [Classical ML Lab](classical-ml.md) | nine scikit-learn algorithms with animated decision regions, trees, clusters and PCA |
| [Neural Lab](neural-lab.md) | configure and train CNNs, LSTMs, transformers, VAEs, GANs, GNNs, diffusion and RL models live |
| [Supported models](supported-models.md) | which model families work where, linking a live run to its graph, and what cannot be exported |
| [Real vision models](vision-benchmarks.md) | speed, GPU/CPU load, accuracy and views of seven pretrained torchvision models, and live GPU fine-tuning |
| [Watching your own training live](live-runs.md) | stream metrics and layer activations from a raw loop or Lightning, Hugging Face Trainer and Keras callback |

## Reference

| Document | Contents |
|---|---|
| [API reference](api.md) | Every REST and WebSocket endpoint: request and response shapes, limits, errors, and project storage |
| [Shared contracts](schemas.md) | The model graph, training event, error and health contracts, and how the Pydantic, JSON Schema (`shared/schemas/`) and Zod versions are kept in sync |
| [Architecture](../ARCHITECTURE.md) | System boundaries, components, data flow and engineering constraints |
| [Development plan](../PLAN.md) | Milestones 0–7 and later, with tasks and acceptance criteria |
| [Skills and standards](../SKILLS.md) | Tools, practices and the contributor checklist |

## Architecture decision records

| ADR | Decision |
|---|---|
| [0001](decisions/0001-2d-renderer.md) | Custom SVG renderer for the neuron-level 2D view |
| [0002](decisions/0002-3d-renderer.md) | 3D view with React Three Fiber, instancing and detail limits |
| [0003](decisions/0003-training-jobs.md) | Training jobs, event streaming and real values |
| [0004](decisions/0004-onnx-import.md) | ONNX import for inspection |
| [0005](decisions/0005-onnx-runtime-inference.md) | Running imported models in an isolated ONNX Runtime worker |
| [0006](decisions/0006-external-runs.md) | External runs from users' own training scripts |
| [0007](decisions/0007-tensor-views.md) | Feature-map, attention and filter views |
| [0008](decisions/0008-run-health-and-controls.md) | Training health alerts and two-way run controls |
| [0009](decisions/0009-live-run-graphs.md) | Link a live PyTorch run to its exported graph |
| [0010](decisions/0010-more-algorithms.md) | Classical ML, Neural Lab and custom Live Run visuals |
| [0011](decisions/0011-framework-callbacks.md) | Lazy framework callbacks, merged steps and pip packaging |
| [0012](decisions/0012-local-project-persistence.md) | Versioned local project persistence without a premature database |
| [0013](decisions/0013-saved-runs-and-comparison.md) | Saved external runs that survive restarts, run comparison and CSV export |
