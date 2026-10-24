"""Demo: a graph neural network predicting which faction each member joins.

Real data, no download: Zachary's karate club (34 members, 78
friendships; ships with networkx, which PyTorch installs). The club split
in two; the model sees the friendship graph but only TWO labels (the
instructor and the administrator) and must infer everyone else's faction
from who is friends with whom. A two-layer graph convolutional network
(GCN) mixes each member's features with their friends'. Live Runs shows the
graph, accuracy over all 34 members, the 2D embedding of every member as it
separates into two groups, and the normalized adjacency matrix the model
uses.

    python sdk/examples/train_gnn.py --url http://127.0.0.1:8001
"""

import time

import torch
from demo_common import arguments, device_for, nv
from torch import nn


def karate() -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import networkx as nx
    except ImportError:
        raise SystemExit("This demo needs networkx: pip install networkx") from None
    graph = nx.karate_club_graph()
    n = graph.number_of_nodes()
    adjacency = torch.zeros(n, n)
    for a, b in graph.edges():
        adjacency[a, b] = adjacency[b, a] = 1.0
    labels = torch.tensor(
        [0 if graph.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)]
    )
    return adjacency, labels


class GCN(nn.Module):
    """H' = act(Â H W), with Â = D^-1/2 (A + I) D^-1/2 (Kipf & Welling)."""

    def __init__(self, adjacency: torch.Tensor, hidden: int = 16) -> None:
        super().__init__()
        a = adjacency + torch.eye(len(adjacency))
        d = a.sum(1).rsqrt()
        self.register_buffer("a_hat", d[:, None] * a * d[None, :])
        n = len(adjacency)
        self.layer1 = nn.Linear(n, hidden)
        self.act1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden, 2)  # A 2D embedding, drawn live.
        self.act2 = nn.Tanh()
        self.classify = nn.Linear(2, 2)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.a_hat @ self.layer1(features))
        return self.act2(self.a_hat @ self.layer2(h))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embed(features))


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=200, lr=0.01)
    device = device_for(args.device)
    torch.manual_seed(0)
    adjacency, labels = karate()
    features = torch.eye(len(adjacency), device=device)  # One-hot identity per member.
    labels = labels.to(device)
    known = torch.tensor([0, 33], device=device)  # Only the two leaders are labeled.
    model = GCN(adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=features)
    with nv.start_run(
        args.name or "GCN on Zachary's karate club",
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else model,
        example_input=features,
        control=True,
        optimizer=optimizer,
    ) as run:
        for epoch in range(1, args.epochs + 1):
            if not run.poll():
                break
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(features)[known], labels[known])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                embedding = model.embed(features)
                accuracy = (model(features).argmax(1) == labels).float().mean().item()
            model.train()
            visuals = {
                "member embeddings (colored by true faction)": nv.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    labels=labels,
                    x_label="e1",
                    y_label="e2",
                )
            }
            if epoch == 1:
                visuals["normalized adjacency Â"] = nv.matrix(
                    model.a_hat,
                    scale="sequential",
                    row_label="member",
                    col_label="member",
                )
            run.log(
                epoch=epoch,
                metrics={"loss": loss.item(), "accuracy (all 34 members)": accuracy},
                health=monitor.collect(),
                visuals=visuals,
            )
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if epoch == 1 or epoch % 40 == 0:
                print(
                    f"epoch {epoch}: loss {loss.item():.4f}, accuracy {accuracy:.3f}",
                    flush=True,
                )
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
