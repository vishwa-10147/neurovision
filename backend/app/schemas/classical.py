"""Classical machine learning in the browser's backend (scikit-learn).

One request fits one algorithm on a small dataset and returns everything the
Classical ML Lab draws: the points, decision regions on a grid for every
animation frame (epochs, trees, depths, k, k-means iterations), and the
algorithm's own structure (support vectors, a decision tree, centroids,
feature importances, principal components). All sizes are bounded.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Algorithm = Literal[
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "knn",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "kmeans",
    "pca",
]
ClassicDataset = Literal["moons", "circles", "spiral", "blobs", "digits"]

GRID = 40
MAX_FRAMES = 40
MAX_TREE_NODES = 127


class ClassicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Algorithm
    dataset: ClassicDataset = "moons"
    samples: Annotated[int, Field(ge=40, le=2000)] = 400
    noise: Annotated[float, Field(ge=0, le=0.5)] = 0.2
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    test_fraction: Annotated[float, Field(ge=0.1, le=0.5)] = 0.25
    # Hyperparameters; each algorithm reads the ones it uses.
    c: Annotated[float, Field(ge=0.001, le=1000)] = 1.0
    gamma: Annotated[float, Field(ge=0.001, le=100)] = 1.0
    neighbors: Annotated[int, Field(ge=1, le=50)] = 15
    max_depth: Annotated[int, Field(ge=1, le=12)] = 5
    estimators: Annotated[int, Field(ge=1, le=200)] = 50
    learning_rate: Annotated[float, Field(ge=0.001, le=1)] = 0.1
    clusters: Annotated[int, Field(ge=1, le=10)] = 3
    epochs: Annotated[int, Field(ge=1, le=200)] = 30


class Points(BaseModel):
    x: list[float]
    y: list[float]
    labels: list[int]


class Frame(BaseModel):
    """One animation step: what changed (e.g. "tree 12") and the regions then."""

    label: str
    # Row-major GRID × GRID cells, bottom row first: the predicted class and
    # its probability (1.0 when the algorithm gives no probabilities).
    grid_class: list[int]
    grid_confidence: list[float]
    train_accuracy: float | None = None
    test_accuracy: float | None = None
    # k-means: centroids and every point's cluster at this iteration.
    centroids: list[tuple[float, float]] | None = None
    assignments: list[int] | None = None
    inertia: float | None = None


class TreeNode(BaseModel):
    id: int
    depth: int
    # Split: feature (0 = x, 1 = y) and threshold; null for a leaf.
    feature: int | None
    threshold: float | None
    left: int | None
    right: int | None
    samples: int
    # Training samples per class reaching the node.
    counts: list[float]
    impurity: float


class PrincipalComponents(BaseModel):
    explained_variance_ratio: list[float]
    # The data projected on the first two components, colored by class.
    projected: Points
    # 2D datasets: component directions from the mean, scaled by std.
    mean: tuple[float, float] | None = None
    axes: list[tuple[float, float]] | None = None


class ClassicResult(BaseModel):
    algorithm: Algorithm
    dataset: ClassicDataset
    classes: int
    features: int
    train: Points
    test: Points
    # Plot area of the regions (2D datasets only).
    bounds: tuple[float, float, float, float] | None
    grid: int
    frames: list[Frame]
    fit_ms: float
    support_vectors: list[tuple[float, float]] | None = None
    tree: list[TreeNode] | None = None
    tree_note: str | None = None
    feature_importance: list[float] | None = None
    pca: PrincipalComponents | None = None
    notes: list[str] = Field(default_factory=list)
