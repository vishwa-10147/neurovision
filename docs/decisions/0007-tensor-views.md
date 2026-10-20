# ADR 0007: Feature-map, attention, and filter views

Status: accepted (Milestone 7)

## Context

People want to see what a CNN's channels and a transformer's attention
heads actually compute for an input. Those tensors are large: one
`[1, 64, 112, 112]` activation holds 800,000 values, and a 512-token,
12-head attention tensor holds 3 million. A run response already returns
only the first 4,096 values of each tensor (ADR 0005), so maps need a
separate, bounded way to read them.

ONNX files say nothing about which tensor is "attention" or which axis is
"channels". Any labels we show are inferences.

## Decision

-   **Keep the last runs' tensors in the worker, not the API.** After a
    run the worker keeps every input and output tensor of the last 4 runs,
    within 256 MB in total (oldest dropped first; a run larger than the
    budget is not kept and the response says so with `retained: false`).
    The API process still never imports ONNX Runtime and never holds full
    tensors. Each run gets a `run_id`; a view of a run that is no longer
    kept, or of a run from before a worker restart, returns
    `410 run_expired` and the UI says to run again.
-   **Two bounded view endpoints** (`shared/schemas/tensor-view.schema.json`):
    -   `GET /models/{id}/maps`: one 2D map, the last two dimensions at
        the given leading indices (`fixed`). Maps over 65,536 cells are
        shrunk by block averaging with a square stride, reported in the
        response; NaN/∞ values are left out of averages and counted.
    -   `GET /models/{id}/grids`: thumbnails along the third-to-last
        dimension (channels, heads): at most 256 items, each at most
        24 × 24 cells after block averaging, plus the overall min/max so
        every thumbnail shares one color scale.
    The downsampling is pure NumPy (`app/ml/tensor_maps.py`) and tested
    against hand-computed block means.
-   **Attention is detected by pattern and labeled as a guess.** A
    `Softmax` over the last axis of a tensor of rank ≥ 3 whose last two
    dimensions are equal (same number or same symbolic name) is reported
    in `report.attention` with a guessed layout (`[batch, heads, query,
    key]` for rank 4). The UI says "detected by pattern" wherever it
    shows one. Verified against a real `torch.nn.MultiheadAttention`
    export: the per-head weights match PyTorch's to 1e-5.
-   **Filters reuse the bounded tensor endpoint.** Conv weights
    `[out, in, kH, kW]` are drawn from the first 4,096 values, at most 16
    filters × 8 input channels, with a note when anything is left out.
-   **Colors.** Signed tensors use a diverging scale (orange negative,
    blue positive) symmetric around zero and shared across the grid.
    Attention uses a fixed 0–1 scale so heads are comparable. Maps are
    canvases with an exact-value readout on hover and with arrow keys.

## Consequences

-   Map views only work for the last few runs; this is stated in the UI.
-   Fused attention operators (`com.microsoft` `Attention`,
    `MultiHeadAttention`), attention whose sequence sizes are unknown,
    and softmaxes on other axes are not detected. They can still be
    viewed as plain tensor maps.
-   Dimension roles for 4D tensors assume NCHW. Channels-last models
    (common from TensorFlow) will show rows × channels maps; the viewer
    states the assumption.
-   No token text: ONNX files carry no tokenizer, so attention axes are
    labeled by position only.
-   Retained tensors raise the worker's memory by up to 256 MB.
