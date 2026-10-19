"""Storage for created models, behind an interface so persistence can change.

The in-memory implementation is for local development: it is per-process,
lost on restart, and evicts the least recently used model past its capacity.
"""

from collections import OrderedDict
from threading import Lock
from typing import Protocol

from app.schemas.graph import ModelGraph


class ModelRepository(Protocol):
    def add(self, graph: ModelGraph) -> None: ...

    def get_graph(self, model_id: str) -> ModelGraph | None: ...


class InMemoryModelRepository:
    def __init__(self, capacity: int = 64) -> None:
        self._capacity = capacity
        self._graphs: OrderedDict[str, ModelGraph] = OrderedDict()
        self._lock = Lock()

    def add(self, graph: ModelGraph) -> None:
        with self._lock:
            self._graphs[graph.model.id] = graph
            self._graphs.move_to_end(graph.model.id)
            while len(self._graphs) > self._capacity:
                self._graphs.popitem(last=False)

    def get_graph(self, model_id: str) -> ModelGraph | None:
        with self._lock:
            graph = self._graphs.get(model_id)
            if graph is not None:
                self._graphs.move_to_end(model_id)
            return graph

    def __len__(self) -> int:
        return len(self._graphs)
