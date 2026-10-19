"""Storage for created models, behind an interface so persistence can change.

The in-memory implementation is for local development: it is per-process,
lost on restart, and evicts the least recently used model past its capacity.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from app.ml.educational_network import EducationalNetwork
from app.schemas.architecture import DenseNetworkConfig
from app.schemas.graph import ModelGraph
from app.schemas.training import EpochMetrics


@dataclass
class ModelRecord:
    graph: ModelGraph
    config: DenseNetworkConfig
    network: EducationalNetwork
    # Held while weights change or are read, so readers never see half a step.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Snapshots from the latest training epoch, keyed like "hidden_1.weight.grad".
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: list[EpochMetrics] = field(default_factory=list)
    last_job_id: str | None = None

    @property
    def model_id(self) -> str:
        return self.graph.model.id


class ModelRepository(Protocol):
    def add(self, record: ModelRecord) -> None: ...

    def get(self, model_id: str) -> ModelRecord | None: ...


class InMemoryModelRepository:
    def __init__(
        self,
        capacity: int = 64,
        on_evict: Callable[[ModelRecord], None] | None = None,
    ) -> None:
        self._capacity = capacity
        self._records: OrderedDict[str, ModelRecord] = OrderedDict()
        self._lock = threading.Lock()
        self._on_evict = on_evict

    def add(self, record: ModelRecord) -> None:
        evicted: list[ModelRecord] = []
        with self._lock:
            self._records[record.model_id] = record
            self._records.move_to_end(record.model_id)
            while len(self._records) > self._capacity:
                evicted.append(self._records.popitem(last=False)[1])
        for old in evicted:
            if self._on_evict:
                self._on_evict(old)

    def get(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            record = self._records.get(model_id)
            if record is not None:
                self._records.move_to_end(model_id)
            return record

    def __len__(self) -> int:
        return len(self._records)
