# ADR 0004: ONNX import for inspection

Status: accepted (Milestone 5)

## Context

Users want to inspect models from real projects. Uploaded model files
are untrusted. Imported graphs are operator graphs, not dense layers of
neurons.

## Decision

-   **Bytes in memory, never executed.** Uploads are read as the raw
    request body: the declared length is checked first, then a streaming
    count enforces the limit. The model is parsed with
    `onnx.load_model_from_string`, which never follows external-data
    paths (verified in a test that fails on any `open` or `stat`). When
    a model references external data, the ONNX checker is skipped
    because it could probe the filesystem. Nothing is executed;
    capabilities report `graph` only.
-   **Validation.** Empty files, unparsable bytes, and protobuf that
    decodes but has no IR version or graph are rejected with
    client-safe messages (`422 invalid_model`). Every string copied from
    the file is truncated, and the operator count is capped.
-   **Normalization.** Non-weight graph inputs become `Input` nodes,
    operators become nodes with generated ids (`node_{i}`), and graph
    outputs become `Output` nodes. Edges follow tensor producers. Weights
    (initializers) are not nodes: they are listed on their consumers
    with generated tensor ids (`t{i}`), since raw names can contain any
    characters. Unknown operators are kept with `unsupported: true` and a
    warning. Placeholder dimensions from shape inference (`unk__N`)
    become `null`.
-   **One repository, two record types.** `ModelRecord` (PyTorch) and
    `ImportedModelRecord` (ONNX) share an LRU repository with a byte
    budget. Forward, training, and metrics endpoints reject imported
    models with `400 capability_unavailable`.
-   **Rendering.** The views draw units only for educational graphs
    (`showsUnits`). For imports, a tensor's last dimension (e.g. image
    width) is not a neuron count, so each operator is one block labeled
    by type, with curved tensor links. A `ModelViewContext` lets the
    Studio and the Explorer each supply their own selection to the same
    2D, 3D, and inspector components.

## Consequences

-   Inference on imported models (Milestone 6) needs a separate
    execution boundary (ONNX Runtime) with time and memory limits; the
    parsing path stays non-executing.
-   Subgraphs are not expanded, and very large graphs need
    level-of-detail or grouping by module path in a later milestone.
