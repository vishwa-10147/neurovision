"""Classical ML Lab: every algorithm fits, animates and describes itself."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.classical import GRID, MAX_FRAMES

client = TestClient(app)


def fit(**body: object) -> dict:
    response = client.post("/api/v1/classical/fit", json=body)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "algorithm",
    [
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "knn",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ],
)
def test_classifiers_draw_regions_and_learn_moons(algorithm: str) -> None:
    result = fit(algorithm=algorithm, dataset="moons", samples=300, noise=0.15)
    assert 1 <= len(result["frames"]) <= MAX_FRAMES
    last = result["frames"][-1]
    assert len(last["grid_class"]) == len(last["grid_confidence"]) == GRID * GRID
    assert set(last["grid_class"]) <= {0, 1}
    assert all(0 <= c <= 1 for c in last["grid_confidence"])
    assert last["test_accuracy"] > 0.8
    assert len(result["train"]["x"]) + len(result["test"]["x"]) == 300


def test_animation_frames_follow_the_algorithm() -> None:
    assert [
        f["label"] for f in fit(algorithm="decision_tree", max_depth=3)["frames"]
    ] == [
        "depth 1",
        "depth 2",
        "depth 3",
    ]
    assert (
        fit(algorithm="random_forest", estimators=5)["frames"][-1]["label"] == "5 trees"
    )
    assert (
        fit(algorithm="logistic_regression", epochs=100)["frames"][-1]["label"]
        == "epoch 100"
    )
    assert (
        len(fit(algorithm="gradient_boosting", estimators=100)["frames"]) == MAX_FRAMES
    )


def test_tree_structure_is_consistent() -> None:
    result = fit(algorithm="decision_tree", dataset="spiral", max_depth=4)
    nodes = {n["id"]: n for n in result["tree"]}
    root = nodes[0]
    assert root["depth"] == 0 and root["samples"] == len(result["train"]["x"])
    for node in nodes.values():
        if node["left"] is not None:
            children = nodes[node["left"]], nodes[node["right"]]
            assert sum(c["samples"] for c in children) == node["samples"]
            assert node["feature"] in (0, 1)
    assert sum(result["feature_importance"]) == pytest.approx(1, abs=1e-3)


def test_svm_reports_support_vectors() -> None:
    result = fit(algorithm="rbf_svm", c=10, gamma=2)
    assert 0 < len(result["support_vectors"]) <= len(result["train"]["x"])


def test_kmeans_iterates_to_convergence_without_labels() -> None:
    result = fit(algorithm="kmeans", dataset="blobs", clusters=4, epochs=50)
    first, last = result["frames"][0], result["frames"][-1]
    assert len(last["centroids"]) == 4
    assert last["inertia"] <= first["inertia"]
    assert len(last["assignments"]) == len(result["train"]["x"])
    assert any("Converged" in n for n in result["notes"])


def test_pca_on_real_digits() -> None:
    result = fit(algorithm="pca", dataset="digits", samples=500)
    pca = result["pca"]
    assert result["features"] == 64 and len(pca["projected"]["x"]) == 500
    ratios = pca["explained_variance_ratio"]
    assert ratios == sorted(ratios, reverse=True) and sum(ratios) <= 1
    assert pca["axes"] is None


def test_pca_on_2d_data_returns_axes() -> None:
    pca = fit(algorithm="pca", dataset="moons")["pca"]
    assert len(pca["axes"]) == 2 and pca["mean"] is not None


def test_high_dimensional_data_without_regions() -> None:
    result = fit(algorithm="knn", dataset="digits", samples=600, neighbors=3)
    assert result["bounds"] is None and result["frames"][-1]["grid_class"] == []
    assert result["frames"][-1]["test_accuracy"] > 0.85


def test_refuses_what_cannot_be_drawn_and_bad_parameters() -> None:
    response = client.post(
        "/api/v1/classical/fit",
        json={"algorithm": "decision_tree", "dataset": "digits"},
    )
    assert response.status_code == 422
    assert "2D dataset" in str(response.json()["error"]["details"])
    for bad in (
        {"algorithm": "svm"},
        {"algorithm": "knn", "neighbors": 0},
        {"algorithm": "pca", "samples": 5000},
    ):
        assert client.post("/api/v1/classical/fit", json=bad).status_code == 422
