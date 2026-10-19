"""Storage for created and imported models, behind an interface.

The in-memory implementation is for local development: it is per-process,
lost on restart, and evicts the least recently used models when it exceeds
its model count or its memory budget (imported models can be large).
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch
from onnx import TensorProto

from app.ml.educational_network import EducationalNetwork
from app.schemas.architecture import DenseNetworkConfig
from app.schemas.graph import ModelGraph
from app.schemas.training import EpochMetrics


@dataclass(kw_only=True)
class StoredModel:
    graph: ModelGraph
    # Held while tensors change or are read, so readers never see half a step.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Approximate memory held by the model, for the repository's budget.
    size_bytes: int = 0

    @property
    def model_id(self) -> str:
        return self.graph.model.id


@dataclass(kw_only=True)
class ModelRecord(StoredModel):
    """An educational network backed by PyTorch."""

    config: DenseNetworkConfig
    network: EducationalNetwork
    # Snapshots from the latest training epoch, keyed like "hidden_1.weight.grad".
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: list[EpochMetrics] = field(default_factory=list)
    last_job_id: str | None = None


@dataclass(kw_only=True)
class ImportedModelRecord(StoredModel):
    """An ONNX model imported for inspection only; it is never executed."""

    # Initializers keyed by the tensor ids used in the graph ("t0", "t1", ...).
    initializers: dict[str, TensorProto] = field(default_factory=dict)


class ModelRepository(Protocol):
    def add(self, record: StoredModel) -> None: ...

    def get(self, model_id: str) -> StoredModel | None: ...


class InMemoryModelRepository:
    def __init__(
        self,
        capacity: int = 64,
        max_bytes: int = 512 * 1024 * 1024,
        on_evict: Callable[[StoredModel], None] | None = None,
    ) -> None:
        self._capacity = capacity
        self._max_bytes = max_bytes
        self._records: OrderedDict[str, StoredModel] = OrderedDict()
        self._lock = threading.Lock()
        self._on_evict = on_evict

    def add(self, record: StoredModel) -> None:
        evicted: list[StoredModel] = []
        with self._lock:
            self._records[record.model_id] = record
            self._records.move_to_end(record.model_id)
            # Evict oldest first, but never the model just added.
            while len(self._records) > 1 and (
                len(self._records) > self._capacity
                or self._total_bytes() > self._max_bytes
            ):
                evicted.append(self._records.popitem(last=False)[1])
        for old in evicted:
            if self._on_evict:
                self._on_evict(old)

    def get(self, model_id: str) -> StoredModel | None:
        with self._lock:
            record = self._records.get(model_id)
            if record is not None:
                self._records.move_to_end(model_id)
            return record

    def _total_bytes(self) -> int:
        return sum(r.size_bytes for r in self._records.values())

    def __len__(self) -> int:
        return len(self._records)
