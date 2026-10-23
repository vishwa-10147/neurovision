"""Demo: a small self-attention model learning a task that needs attention.

Task (generated, no download): sequences of 12 tokens from a vocabulary of
16; the label is 1 when the first token appears again later in the
sequence. Answering needs every position to look back at the first one,
which is what self-attention does. Live Runs shows the network's graph
(embedding, multi-head attention, feed-forward) colored by each layer's
activations as it learns; the trained attention weights can then be viewed
per head in the Model Explorer.

    python sdk/examples/train_attention.py --url http://127.0.0.1:8001
"""

import torch
from demo_common import arguments, train_live
from torch import nn

VOCAB, LENGTH = 16, 12


class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, written out step by step.

    Written by hand rather than with nn.MultiheadAttention, whose fused
    inference kernel cannot be exported to ONNX; here every step (projections,
    scores, softmax, weighted sum) is its own operator in the graph.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(dim, dim)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed sizes (-1 for the batch) export as one Reshape, without the
        # shape arithmetic that x.shape-based sizes add to the ONNX graph.
        return x.reshape(-1, LENGTH, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (
            self.split(self.query(x)),
            self.split(self.key(x)),
            self.split(self.value(x)),
        )
        # [batch, heads, query, key]
        weights = self.softmax(q @ k.transpose(-2, -1) / self.head_dim**0.5)
        mixed = (
            (weights @ v)
            .transpose(1, 2)
            .reshape(-1, LENGTH, self.heads * self.head_dim)
        )
        return self.out(mixed)


class AttentionClassifier(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, dim)
        self.position = nn.Parameter(torch.randn(1, LENGTH, dim) * 0.1)
        self.attention = SelfAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.position
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.feed_forward(x))
        return self.head(x.mean(dim=1))


def sequences(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced: half the sequences repeat their first token later."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (count, LENGTH), generator=generator)
    label = torch.arange(count) % 2
    for i in range(count):
        first = tokens[i, 0]
        rest = tokens[i, 1:]
        # Remove the first token from the rest, then add it back for label 1.
        others = torch.tensor([t for t in range(VOCAB) if t != first])
        repeats = rest == first
        rest[repeats] = others[
            torch.randint(0, VOCAB - 1, (int(repeats.sum()),), generator=generator)
        ]
        if label[i] == 1:
            rest[torch.randint(0, LENGTH - 1, (1,), generator=generator)] = first
    return tokens, label


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=40, lr=0.002)
    torch.manual_seed(0)
    x_train, y_train = sequences(4000, seed=0)
    x_val, y_val = sequences(1000, seed=1)
    train_live(
        "Attention: is the first token repeated?",
        AttentionClassifier(),
        (x_train, y_train, x_val, y_val),
        args=args,
        probe_layers=["attention.softmax", "norm1", "feed_forward.1", "norm2"],
    )


if __name__ == "__main__":
    main()
