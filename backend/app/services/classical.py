"""Fit classical scikit-learn algorithms and describe them for drawing.

Every number returned is computed from the fitted models; animation frames
come from the algorithms' own intermediate states (SGD epochs, staged
boosting predictions, the first n trees of a forest, k-means iterations).
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.ml.datasets import make_dataset
from app.schemas.classical import (
    GRID,
    MAX_FRAMES,
    MAX_TREE_NODES,
    ClassicRequest,
    ClassicResult,
    Frame,
    Points,
    PrincipalComponents,
    TreeNode,
)

SPIRAL_CLASSES = 3


class ClassicError(ValueError):
    """The request cannot be fitted (e.g. an algorithm needs 2D data)."""


def _data(
    request: ClassicRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if request.dataset in ("moons", "circles", "spiral"):
        classes = SPIRAL_CLASSES if request.dataset == "spiral" else 2
        data = make_dataset(
            request.dataset,
            samples=request.samples,
            noise=request.noise,
            input_units=2,
            output_units=classes,
            validation_split=request.test_fraction,
            seed=request.seed,
        )
        return (
            data.x_train.numpy().astype(np.float64),
            data.y_train.numpy(),
            data.x_val.numpy().astype(np.float64),
            data.y_val.numpy(),
            classes,
        )
    rng = np.random.default_rng(request.seed)
    if request.dataset == "blobs":
        x, y = make_blobs(
            n_samples=request.samples,
            centers=4,
            cluster_std=0.6 + 2 * request.noise,
            random_state=request.seed,
        )
        classes = 4
    else:  # digits: 64 features, 10 classes, real images.
        digits = load_digits()
        index = rng.permutation(len(digits.target))[: request.samples]
        x, y = digits.data[index] / 16.0, digits.target[index]
        classes = 10
    order = rng.permutation(len(x))
    split = int(len(x) * (1 - request.test_fraction))
    train, test = order[:split], order[split:]
    return x[train], y[train], x[test], y[test], classes


def _points(x: np.ndarray, y: np.ndarray) -> Points:
    return Points(
        x=np.round(x[:, 0], 4).tolist(),
        y=np.round(x[:, 1], 4).tolist(),
        labels=[int(v) for v in y],
    )


def _bounds(x: np.ndarray) -> tuple[float, float, float, float]:
    low, high = x.min(axis=0), x.max(axis=0)
    pad = (high - low) * 0.08 + 1e-6
    return (
        float(low[0] - pad[0]),
        float(high[0] + pad[0]),
        float(low[1] - pad[1]),
        float(high[1] + pad[1]),
    )


def _grid_points(bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    xs, ys = np.linspace(x0, x1, GRID), np.linspace(y0, y1, GRID)
    gx, gy = np.meshgrid(xs, ys)  # Row 0 is the bottom (lowest y).
    return np.column_stack([gx.ravel(), gy.ravel()])


def _frame(
    label: str,
    probabilities: np.ndarray,
    classes_seen: np.ndarray,
    accuracies: tuple[float, float] | None = None,
) -> Frame:
    best = probabilities.argmax(axis=1)
    return Frame(
        label=label,
        grid_class=[int(classes_seen[i]) for i in best],
        grid_confidence=np.round(probabilities.max(axis=1), 2).tolist(),
        train_accuracy=None if accuracies is None else accuracies[0],
        test_accuracy=None if accuracies is None else accuracies[1],
    )


def _hard(predictions: np.ndarray, classes: int) -> np.ndarray:
    """Class predictions as one-hot 'probabilities' (no confidence available)."""
    out = np.zeros((len(predictions), classes))
    out[np.arange(len(predictions)), predictions] = 1.0
    return out


def _thin(items: list, limit: int = MAX_FRAMES) -> list:
    """At most `limit` evenly spaced items, always keeping the last."""
    if len(items) <= limit:
        return items
    picks = np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))
    return [items[i] for i in picks]


def _tree(model: DecisionTreeClassifier) -> list[TreeNode]:
    tree = model.tree_
    nodes: list[TreeNode] = []
    queue = [(0, 0)]
    while queue and len(nodes) < MAX_TREE_NODES:
        node, depth = queue.pop(0)
        leaf = tree.children_left[node] == -1
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        counts = tree.value[node][0] * tree.n_node_samples[node]  # Fractions → counts.
        nodes.append(
            TreeNode(
                id=node,
                depth=depth,
                feature=None if leaf else int(tree.feature[node]),
                threshold=None if leaf else float(tree.threshold[node]),
                left=None if leaf else left,
                right=None if leaf else right,
                samples=int(tree.n_node_samples[node]),
                counts=np.round(counts, 2).tolist(),
                impurity=float(tree.impurity[node]),
            )
        )
        if not leaf:
            queue += [(left, depth + 1), (right, depth + 1)]
    kept = {n.id for n in nodes}
    for n in nodes:  # Children cut by the size limit are shown as missing.
        if n.left is not None and n.left not in kept:
            n.left = n.right = None
    return nodes


def fit(request: ClassicRequest) -> ClassicResult:
    x_train, y_train, x_test, y_test, classes = _data(request)
    two_d = x_train.shape[1] == 2
    algorithm = request.algorithm
    if not two_d and algorithm not in (
        "pca",
        "kmeans",
        "logistic_regression",
        "random_forest",
        "knn",
    ):
        raise ClassicError(
            f"{algorithm} is drawn as decision regions, which needs a 2D dataset; "
            "use moons, circles, spiral or blobs (digits works with PCA)."
        )
    if algorithm == "kmeans" and not two_d:
        raise ClassicError(
            "k-means is drawn in 2D; use moons, circles, spiral or blobs."
        )
    bounds = _bounds(np.vstack([x_train, x_test])) if two_d else None
    grid = _grid_points(bounds) if bounds else None
    notes: list[str] = []
    result = ClassicResult(
        algorithm=algorithm,
        dataset=request.dataset,
        classes=classes,
        features=x_train.shape[1],
        train=_points(x_train, y_train) if two_d else Points(x=[], y=[], labels=[]),
        test=_points(x_test, y_test) if two_d else Points(x=[], y=[], labels=[]),
        bounds=bounds,
        grid=GRID,
        frames=[],
        fit_ms=0.0,
    )
    started = time.perf_counter()

    def accuracy(model) -> tuple[float, float]:
        return float(model.score(x_train, y_train)), float(model.score(x_test, y_test))

    if algorithm in ("logistic_regression", "linear_svm"):
        scaler = StandardScaler().fit(x_train)
        xs_train, xs_test = scaler.transform(x_train), scaler.transform(x_test)
        model = SGDClassifier(
            loss="log_loss" if algorithm == "logistic_regression" else "hinge",
            alpha=1.0 / (request.c * len(x_train)),
            learning_rate="optimal",
            random_state=request.seed,
        )
        labels = np.arange(classes)
        frames = []
        for epoch in range(1, request.epochs + 1):
            model.partial_fit(xs_train, y_train, classes=labels)
            accs = (
                float(model.score(xs_train, y_train)),
                float(model.score(xs_test, y_test)),
            )
            if grid is not None:
                g = scaler.transform(grid)
                probs = (
                    model.predict_proba(g)
                    if algorithm == "logistic_regression"
                    else _hard(model.predict(g), classes)
                )
                frames.append(_frame(f"epoch {epoch}", probs, labels, accs))
            else:
                frames.append(
                    Frame(
                        label=f"epoch {epoch}",
                        grid_class=[],
                        grid_confidence=[],
                        train_accuracy=accs[0],
                        test_accuracy=accs[1],
                    )
                )
        result.frames = _thin(frames)
        if algorithm == "linear_svm":
            notes.append(
                "Linear SVM trained with stochastic gradient descent on the hinge loss; "
                "the regions show the predicted class only."
            )
    elif algorithm == "rbf_svm":
        # Probabilities by Platt scaling on held-out folds of the training set.
        model = CalibratedClassifierCV(
            SVC(C=request.c, gamma=request.gamma, random_state=request.seed),
            ensemble=False,
        ).fit(x_train, y_train)
        svm = model.calibrated_classifiers_[0].estimator
        result.frames = [
            _frame("fitted", model.predict_proba(grid), model.classes_, accuracy(model))
        ]
        result.support_vectors = [
            tuple(map(float, v)) for v in svm.support_vectors_[:500]
        ]
        notes.append(
            f"{len(svm.support_vectors_)} support vectors of {len(x_train)} training points."
        )
    elif algorithm == "knn":
        ks = list(range(1, request.neighbors + 1, 2 if request.neighbors > 10 else 1))
        if ks[-1] != request.neighbors:
            ks.append(request.neighbors)
        frames = []
        for k in _thin(ks):
            model = KNeighborsClassifier(n_neighbors=min(k, len(x_train))).fit(
                x_train, y_train
            )
            if grid is not None:
                frames.append(
                    _frame(
                        f"k = {k}",
                        model.predict_proba(grid),
                        model.classes_,
                        accuracy(model),
                    )
                )
            else:
                accs = accuracy(model)
                frames.append(
                    Frame(
                        label=f"k = {k}",
                        grid_class=[],
                        grid_confidence=[],
                        train_accuracy=accs[0],
                        test_accuracy=accs[1],
                    )
                )
        result.frames = frames
    elif algorithm == "decision_tree":
        frames = []
        model = None
        for depth in range(1, request.max_depth + 1):
            model = DecisionTreeClassifier(
                max_depth=depth, random_state=request.seed
            ).fit(x_train, y_train)
            frames.append(
                _frame(
                    f"depth {depth}",
                    model.predict_proba(grid),
                    model.classes_,
                    accuracy(model),
                )
            )
        result.frames = frames
        result.tree = _tree(model)
        result.feature_importance = model.feature_importances_.round(4).tolist()
        if model.tree_.node_count > MAX_TREE_NODES:
            result.tree_note = (
                f"Showing the first {MAX_TREE_NODES} of {model.tree_.node_count} nodes."
            )
    elif algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=request.estimators,
            max_depth=request.max_depth,
            random_state=request.seed,
        ).fit(x_train, y_train)
        frames = []
        for n in _thin(list(range(1, request.estimators + 1))):
            trees = model.estimators_[:n]

            def proba(points: np.ndarray, trees=trees) -> np.ndarray:
                return np.mean([t.predict_proba(points) for t in trees], axis=0)

            accs = (
                float((model.classes_[proba(x_train).argmax(1)] == y_train).mean()),
                float((model.classes_[proba(x_test).argmax(1)] == y_test).mean()),
            )
            if grid is not None:
                frames.append(
                    _frame(
                        f"{n} tree{'s' if n > 1 else ''}",
                        proba(grid),
                        model.classes_,
                        accs,
                    )
                )
            else:
                frames.append(
                    Frame(
                        label=f"{n} trees",
                        grid_class=[],
                        grid_confidence=[],
                        train_accuracy=accs[0],
                        test_accuracy=accs[1],
                    )
                )
        result.frames = frames
        result.feature_importance = model.feature_importances_.round(4).tolist()
        if two_d:
            result.tree = _tree(model.estimators_[0])
            result.tree_note = (
                "The forest's first tree (each tree sees a bootstrap sample)."
            )
    elif algorithm == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=request.estimators,
            learning_rate=request.learning_rate,
            max_depth=min(request.max_depth, 5),
            random_state=request.seed,
        ).fit(x_train, y_train)
        stages = list(model.staged_predict_proba(grid))
        train_stages = list(model.staged_predict(x_train))
        test_stages = list(model.staged_predict(x_test))
        picks = _thin(list(range(len(stages))))
        result.frames = [
            _frame(
                f"{i + 1} tree{'s' if i else ''}",
                stages[i],
                model.classes_,
                (
                    float((train_stages[i] == y_train).mean()),
                    float((test_stages[i] == y_test).mean()),
                ),
            )
            for i in picks
        ]
        result.feature_importance = model.feature_importances_.round(4).tolist()
    elif algorithm == "kmeans":
        rng = np.random.RandomState(request.seed)
        x_all = np.vstack([x_train, x_test])
        centers, _ = kmeans_plusplus(
            x_all, n_clusters=request.clusters, random_state=rng
        )
        frames = []
        for iteration in range(1, request.epochs + 1):
            distances = ((x_all[:, None, :] - centers[None]) ** 2).sum(-1)
            assign = distances.argmin(1)
            inertia = float(distances.min(1).sum())
            region = ((grid[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
            frame = _frame(
                f"iteration {iteration}",
                _hard(region, request.clusters),
                np.arange(request.clusters),
            )
            frame.centroids = [tuple(map(float, c)) for c in centers]
            frame.assignments = assign.tolist()
            frame.inertia = inertia
            frames.append(frame)
            new = np.array(
                [
                    x_all[assign == k].mean(0) if (assign == k).any() else centers[k]
                    for k in range(request.clusters)
                ]
            )
            if np.allclose(new, centers):
                notes.append(f"Converged after {iteration} iterations.")
                break
            centers = new
        result.frames = _thin(frames)
        result.train = _points(x_all, np.concatenate([y_train, y_test]))
        result.test = Points(x=[], y=[], labels=[])
        notes.append(
            "k-means never sees the labels; point colors show the clusters it found."
        )
    else:  # pca
        x_all = np.vstack([x_train, x_test])
        y_all = np.concatenate([y_train, y_test])
        model = PCA().fit(x_all)
        projected = model.transform(x_all)[:, :2]
        result.pca = PrincipalComponents(
            explained_variance_ratio=model.explained_variance_ratio_[:10]
            .round(5)
            .tolist(),
            projected=Points(
                x=projected[:, 0].tolist(),
                y=projected[:, 1].tolist(),
                labels=y_all.tolist(),
            ),
            mean=tuple(map(float, model.mean_)) if two_d else None,
            axes=[
                tuple(map(float, c * np.sqrt(v)))
                for c, v in zip(
                    model.components_, model.explained_variance_, strict=True
                )
            ]
            if two_d
            else None,
        )
        if not two_d:
            notes.append(
                f"{x_all.shape[1]} features reduced to 2; the first two components keep "
                f"{model.explained_variance_ratio_[:2].sum():.0%} of the variance."
            )
    result.fit_ms = round((time.perf_counter() - started) * 1000, 1)
    result.notes = notes
    return result
