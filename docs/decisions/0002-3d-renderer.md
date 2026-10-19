# ADR 0002: 3D view with React Three Fiber, instancing, and detail limits

Status: accepted (Milestone 3)

## Context

The Architecture Studio needs a 3D view of the same normalized graph as
the 2D view, with selection synchronized between both views and the
inspector. It must stay responsive and say clearly when a graph is too
large to draw in full detail.

## Decision

-   **Planning is separate from drawing.** `planScene(graph)` in
    `frontend/src/features/visualization-3d/scene.ts` is a pure function.
    It produces unit positions (each layer is a near-square grid in the
    y-z plane, layers spaced along x), connection pairs, layer links,
    detail flags, and bounds. `highlightFor(scene, selection)` turns the
    shared selection into per-instance emphasis. Both are unit-tested
    without WebGL.
-   **Batched geometry.** All units are one `InstancedMesh`, and all
    unit-to-unit connections are one `LineSegments` with vertex colors.
    That is two draw calls however large the network is. Rendering is
    on demand (`frameloop="demand"`).
-   **Detail limits.**
    -   More than **2,048 units**: layers are drawn as solid blocks.
    -   More than **8,000 connections**: one link per layer pair.
    -   Unknown layer sizes (e.g. imported graphs): blocks and links;
        units are never invented.

    The legend states which limit applied. The editor's largest network
    (16 → 6 × 32 → 10, 5,952 connections) stays at full detail.
-   **Shared selection.** Clicks set the same semantic `Selection` in
    the Zustand store as the 2D view does. The view mode lives in the
    store too, so switching views keeps the selection and model.
-   **Framing.** `defaultCamera` binary-searches the camera distance
    until every corner of the scene's bounding box (layer labels
    included) projects inside the view frustum, for the canvas's real
    aspect ratio. A bounding sphere alone wasted space on long networks
    or clipped the end nearest the camera. The view refits on resize
    until the user moves the camera. Clip planes follow the framing
    distance.
-   **Containment.** The 3D view sits in an error boundary, so a WebGL or
    chunk-loading failure shows a message and a switch back to 2D
    instead of blanking the app.
-   **Camera.** drei `OrbitControls` handles mouse orbit, zoom, and pan.
    Toolbar buttons and keyboard shortcuts (arrows orbit, Shift+arrows
    pan, +/− zoom, 0 reset, Esc clears the selection) drive the same
    camera. Layer buttons below the canvas give keyboard access to
    selection. Damping is turned off for `prefers-reduced-motion`. No
    signal animation exists yet, because no measured activations exist
    yet.
-   **Loading cost.** three.js and R3F (~950 kB, ~250 kB gzipped) are
    lazy-loaded on first switch to 3D. Without WebGL, the view explains
    this and offers a switch back to 2D.

## Consequences

-   Large imported graphs (Milestone 5) will render as blocks and links
    by default. Drill-down into a selected layer is future work.
-   Layer labels use DOM overlays (`Html`), which is fine for tens of
    layers. Hundreds of operators will need a different labeling
    strategy.
