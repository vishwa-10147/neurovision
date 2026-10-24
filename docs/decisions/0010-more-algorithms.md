# ADR 0010: Classical ML Lab, Neural Lab, custom visuals, scikit-learn models

Status: accepted

## Context

NeuroVision covered dense networks (Studio), imported ONNX models
(Explorer) and the user's own PyTorch scripts (Live Runs). Users asked for
more algorithms: classical machine learning, more neural architectures
(CNN, RNN, transformer, autoencoder, VAE, GAN, GNN, diffusion,
reinforcement learning) to configure and watch, and scikit-learn models in
the Explorer.

## Decision

1.  **Custom visuals on live steps.** A logged step may carry up to six
    named visuals: a scatter plot (≤ 512 points, optional class per point)
    or a matrix (≤ 64 × 64, diverging or sequential scale). The SDK adds
    `nv.scatter`, `nv.matrix` and `nv.image_grid`; batches are split to
    stay under the 2 MB body limit; only the latest visuals are kept in
    the replay history. One general mechanism serves latent spaces, hidden
    states, attention, generated images and RL trajectories.
2.  **Classical ML Lab: synchronous scikit-learn fits.** `POST
    /classical/fit` fits one of nine algorithms (≤ 2,000 samples) in the
    thread pool and returns regions on a 40 × 40 grid for up to 40
    animation frames taken from the algorithm's own intermediate states
    (SGD epochs, the first n trees, staged boosting, k-means iterations),
    plus trees, support vectors, centroids or principal components.
    Fits take 4–400 ms, so no job machinery is needed. SVM probabilities
    use `CalibratedClassifierCV` (`SVC(probability=True)` is deprecated
    in scikit-learn 1.9).
3.  **Neural Lab launches NeuroVision's own scripts.** Rather than a
    second, in-server training engine, the lab starts the bundled
    `sdk/examples` scripts as processes that stream through the SDK
    exactly like user code, so graphs, health, visuals, system figures and
    controls come for free and stay identical. Only listed scripts run;
    parameters are validated against typed, bounded templates and passed
    as separate arguments; at most two run at once; the lab is on by
    default only in development; the interpreter is configurable (a CUDA
    build for the GPU).
4.  **scikit-learn models in the Explorer.** ONNX-ML operators are
    standard ONNX and already ran in ONNX Runtime. Two gaps were closed:
    the worker crashed on non-tensor outputs (skl2onnx's default ZipMap
    returns a list of maps); they are now converted to a `[batch,
    classes]` tensor with a note. Tree ensembles are parsed from their
    full attribute arrays (node attributes are truncated for display) into
    the first three trees, drawn with the same tree view as the Classical
    ML Lab.

Findings from running everything, fixed here:

-   PyTorch 2.14's `optimizer.step()` initializes CUDA even for CPU
    models, so "CUDA initialized" no longer implies "training on the GPU".
    The SDK now reports the device of the linked model's parameters,
    which keeps `gpu_underused` from firing for CPU training.
-   Rules for `loss` do not fit policy-gradient losses; the RL demo logs
    `policy_loss`.
-   `nn.MultiheadAttention`'s fused inference kernel cannot be exported
    to ONNX; the transformer demo writes attention out.

## Consequences

-   One SDK call (`visuals=`) covers model-specific views without new
    event types; the UI draws whatever the script measured, as sent.
-   The lab cannot run arbitrary code, but it does start processes: hosts
    that are not development machines must opt in.
-   Classical fits are synchronous and bounded; larger data belongs in a
    script with Live Runs.
