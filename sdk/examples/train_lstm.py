"""Demo: an LSTM forecasting a time series, live.

Data (generated, no download): a signal made of two seasonal waves, a slow
trend and noise, like daily demand. The LSTM reads 30 past values and
predicts the next one. Live Runs shows the graph, the loss, and two
visuals: the forecast against the real values on held-out data, and the
LSTM's hidden state at every time step (time × unit) for one window.

    python sdk/examples/train_lstm.py --url http://127.0.0.1:8001
"""

import argparse
import math
import time

import torch
from demo_common import arguments, device_for, nv
from torch import nn

WINDOW = 30


CELLS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}


class Forecaster(nn.Module):
    """A recurrent layer stack (LSTM, GRU or plain RNN) and a linear head."""

    def __init__(self, hidden: int = 32, layers: int = 1, cell: str = "lstm") -> None:
        super().__init__()
        self.rnn = CELLS[cell](1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.rnn(x)  # [batch, time, hidden]
        return self.head(states[:, -1])


def signal(length: int = 3000) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    series = (
        torch.sin(2 * math.pi * t / 50)
        + 0.5 * torch.sin(2 * math.pi * t / 7)
        + 0.0005 * t
        + 0.1 * torch.randn(length, generator=generator)
    )
    return (series - series.mean()) / series.std()


def windows(series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = series.unfold(0, WINDOW, 1)[:-1].unsqueeze(-1)  # [n, WINDOW, 1]
    y = series[WINDOW:].unsqueeze(-1)
    return x, y


def main() -> None:
    def options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--cell", choices=sorted(CELLS), default="lstm")
        parser.add_argument("--hidden", type=int, default=32, help="hidden units")
        parser.add_argument(
            "--layers", type=int, default=1, help="stacked recurrent layers"
        )

    args = arguments(__doc__.splitlines()[0], epochs=30, lr=0.003, extra=options)
    device = device_for(args.device)
    torch.manual_seed(0)
    x, y = windows(signal())
    split = int(len(x) * 0.8)  # Train on the past, test on the future.
    x_train, y_train, x_val, y_val = (
        t.to(device) for t in (x[:split], y[:split], x[split:], y[split:])
    )
    model = Forecaster(args.hidden, args.layers, args.cell).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])
    loss_fn = nn.MSELoss()
    step = 0
    with nv.start_run(
        args.name or f"{args.cell.upper()} forecaster",
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else model,
        example_input=x_val[:1],
        control=True,
        optimizer=optimizer,
    ) as run:
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            seen = 0
            for index in torch.randperm(len(x_train), device=device).split(args.batch):
                if not run.poll():
                    break
                optimizer.zero_grad()
                loss = loss_fn(model(x_train[index]), y_train[index])
                loss.backward()
                optimizer.step()
                step += 1
                total += loss.item() * len(index)
                seen += len(index)
            if seen == 0:
                break
            model.eval()
            with torch.no_grad():
                forecast = model(x_val)
                val_loss = loss_fn(forecast, y_val).item()
                states, _ = model.rnn(x_val[:1])  # Last layer's state per time step.
            model.train()
            stretch = 120
            run.log(
                epoch=epoch,
                step=step,
                metrics={"loss": total / seen, "val_loss": val_loss},
                health=monitor.collect(),
                samples=seen,
                visuals={
                    "forecast vs actual (held out)": nv.scatter(
                        list(range(stretch)) * 2,
                        torch.cat([y_val[:stretch, 0], forecast[:stretch, 0]]),
                        labels=[0] * stretch + [1] * stretch,
                        x_label="time step (blue = actual, red = forecast)",
                        y_label="value",
                    ),
                    f"{args.cell.upper()} hidden state": nv.matrix(
                        states[0], row_label="time step", col_label="hidden unit"
                    ),
                },
            )
            if run.should_stop:
                break
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"epoch {epoch}: loss {total / seen:.4f} val_loss {val_loss:.4f}",
                    flush=True,
                )
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
