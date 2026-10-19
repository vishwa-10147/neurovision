# ADR 0001: Custom SVG renderer for the neuron-level 2D view

Status: accepted (Milestone 2)

## Context

The Architecture Studio shows each unit (neuron) and each unit-to-unit
connection of a small dense network, with selection synchronized with
the inspector and, from Milestone 3, with the 3D view. ARCHITECTURE.md
suggests React Flow for layer-level editing and a custom SVG or Canvas
view for neuron-level diagrams.

## Decision

-   The neuron-level view is a custom SVG renderer
    (`frontend/src/features/visualization-2d/`). The layer editor is a
    form. React Flow is not used yet: there is no free-form graph
    editing to justify the dependency.
-   Layout is a pure function of the normalized graph (`computeLayout`).
    It is never stored in the graph.
-   Selection is held in a Zustand store as semantic ids (`layer`,
    `unit` with an index, or `connection` with an edge id and optional
    unit indices). It is not tied to SVG elements.
-   Above **1,500** unit-to-unit connections, each layer pair is drawn as
    one band instead of individual lines. This bounds the DOM to about
    1,500 lines plus at most 6 × 32 + 26 units. The legend states when
    this happens.
-   Architecture limits (input ≤ 16, hidden ≤ 32 units × 6 layers,
    output ≤ 10) are enforced by both the backend and the frontend.

## Consequences

-   Rendering stays responsive without Canvas or WebGL for every network
    the editor allows.
-   Imported models whose layers have unknown sizes are drawn as blocks,
    with bands for edges, rather than with invented units.
-   If larger graphs or free-form editing are needed (for example ONNX
    graphs with hundreds of operators), revisit: React Flow for
    operator-level graphs, or Canvas for dense neuron views.
