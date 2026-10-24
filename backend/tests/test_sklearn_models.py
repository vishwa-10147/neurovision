"""scikit-learn models exported with skl2onnx: imported, drawn, and run.

ONNX-ML operators (TreeEnsembleClassifier, LinearClassifier, SVMClassifier,
…) are standard ONNX and run in ONNX Runtime; tree ensembles also carry their
first trees for the inspector's tree view.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.main import create_app

skl2onnx = pytest.importorskip("skl2onnx")

X, Y = load_iris(return_X_y=True)
X = X.astype(np.float32)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def imported(client: TestClient, model, zipmap: bool = False) -> dict:
    model.fit(X, Y)
    options = None if zipmap else {"zipmap": False}
    onnx_model = skl2onnx.to_onnx(model, X[:1], options=options)
    response = client.post(
        "/api/v1/import/onnx?filename=model.onnx",
        content=onnx_model.SerializeToString(),
    )
    assert response.status_code == 201, response.text
    return response.json()


def run(client: TestClient, result: dict) -> dict:
    name = result["report"]["inputs"][0]["name"]
    response = client.post(
        f"/api/v1/models/{result['model_id']}/run",
        json={
            "inputs": {
                name: {"shape": [3, 4], "values": X[[0, 60, 120]].flatten().tolist()}
            }
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(max_depth=3, random_state=0),
        RandomForestClassifier(5, random_state=0),
        LogisticRegression(max_iter=500),
    ],
)
def test_runs_and_matches_scikit_learn(client: TestClient, model) -> None:
    result = imported(client, model)
    assert result["report"]["unsupported_operators"] == []
    outputs = {o["name"]: o for o in run(client, result)["outputs"]}
    assert outputs["label"]["values"] == model.predict(X[[0, 60, 120]]).tolist()
    probabilities = np.array(outputs["probabilities"]["values"]).reshape(3, 3)
    assert np.allclose(probabilities, model.predict_proba(X[[0, 60, 120]]), atol=1e-5)


def test_tree_ensembles_carry_their_trees(client: TestClient) -> None:
    model = GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0)
    result = imported(client, model)
    node = next(
        n for n in result["graph"]["nodes"] if n["op_type"] == "TreeEnsembleClassifier"
    )
    ensemble = node["metadata"]["onnx"]["tree_ensemble"]
    assert (
        ensemble["kind"] == "classifier" and ensemble["trees"] == 4 * 3
    )  # One tree per class per stage.
    first = ensemble["shown"][0]
    root = first[0]
    assert root["depth"] == 0 and root["mode"] == "BRANCH_LEQ"
    reference = model.estimators_[0][0].tree_
    assert root["feature"] == reference.feature[0]
    assert root["threshold"] == pytest.approx(reference.threshold[0], rel=1e-6)
    leaves = [n for n in first if n["mode"] == "LEAF"]
    assert leaves and all(n["left"] is None and len(n["values"]) == 3 for n in leaves)
    assert len(ensemble["shown"]) == 3  # At most three trees are shown.


def test_zipmap_outputs_are_converted_with_a_note(client: TestClient) -> None:
    result = imported(client, SVC(), zipmap=True)
    body = run(client, result)
    outputs = {o["name"]: o for o in body["outputs"]}
    assert outputs["output_probability"]["shape"] == [3, 3]
    assert any("ZipMap" in note for note in body["notes"])
