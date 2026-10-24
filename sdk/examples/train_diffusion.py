"""Demo: a tiny diffusion model learning to generate a spiral, live.

Diffusion models (the idea behind image generators) learn to remove noise.
Training adds a random amount of Gaussian noise to points of a 2D spiral
(generated, no download) and teaches a small network to predict that
noise. Sampling then starts from pure noise and removes it step by step
(DDPM, 100 steps). Live Runs shows the denoiser's graph, the loss, and 400
freshly generated points every 10 epochs: a blob that turns into a spiral
(it takes a few hundred epochs; the loss barely moves, the samples do).

    python sdk/examples/train_diffusion.py --url http://127.0.0.1:8001
"""

import math
import time

import torch
from demo_common import arguments, device_for, nv
from torch import nn

STEPS = 100


class Denoiser(nn.Module):
    """Input: noisy point (x, y) and the noise level t/STEPS; output: the noise.

    The noise level enters as sine/cosine features at several frequencies,
    as in diffusion models generally, so the network can tell levels apart.
    """

    def __init__(self, hidden: int = 128, frequencies: int = 8) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(frequencies) * math.pi)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * frequencies, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )  # fmt: skip

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level = inputs[:, 2:3] * self.frequencies
        return self.net(
            torch.cat([inputs[:, :2], torch.sin(level), torch.cos(level)], dim=1)
        )


def spiral(n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    t = torch.rand(n, generator=generator) * 3 * math.pi + 0.5
    points = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) / 5
    return points + 0.04 * torch.randn(n, 2, generator=generator)


@torch.no_grad()
def sample(
    model: nn.Module,
    n: int,
    alphas: torch.Tensor,
    bars: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    x = torch.randn(n, 2, device=alphas.device)
    for t in reversed(range(STEPS)):
        level = torch.full((n, 1), t / STEPS, device=x.device)
        noise = model(torch.cat([x, level], dim=1))
        x = (x - betas[t] / torch.sqrt(1 - bars[t]) * noise) / torch.sqrt(alphas[t])
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
    return x


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=300, lr=0.001)
    device = device_for(args.device)
    torch.manual_seed(0)
    data = spiral(8000).to(device)
    betas = torch.linspace(1e-4, 0.05, STEPS, device=device)
    alphas = 1 - betas
    bars = torch.cumprod(alphas, dim=0)
    model = Denoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    probe = torch.cat([data[:64], torch.full((64, 1), 0.5, device=device)], dim=1)
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=probe)
    step = 0
    with nv.start_run(
        args.name or "Diffusion model drawing a spiral",
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else model,
        example_input=probe[:1],
        control=True,
        optimizer=optimizer,
    ) as run:
        for epoch in range(1, args.epochs + 1):
            total, batches = 0.0, 0
            for index in torch.randperm(len(data), device=device).split(args.batch * 2):
                if not run.poll():
                    break
                x0 = data[index]
                t = torch.randint(0, STEPS, (len(x0),), device=device)
                noise = torch.randn_like(x0)
                xt = (
                    torch.sqrt(bars[t])[:, None] * x0
                    + torch.sqrt(1 - bars[t])[:, None] * noise
                )
                predicted = model(torch.cat([xt, (t / STEPS)[:, None].float()], dim=1))
                loss = nn.functional.mse_loss(predicted, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                batches += 1
                total += loss.item()
            if batches == 0:
                break
            visuals = {}
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                model.eval()
                points = sample(model, 400, alphas, bars, betas).clamp(-4, 4)
                model.train()
                visuals["generated points"] = nv.scatter(points[:, 0], points[:, 1])
                if epoch == 1:
                    visuals["training data"] = nv.scatter(data[:400, 0], data[:400, 1])
            run.log(
                epoch=epoch,
                step=step,
                metrics={"loss": total / batches},
                health=monitor.collect(),
                samples=len(data),
                visuals=visuals,
            )
            if run.should_stop:
                break
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if epoch == 1 or epoch % 50 == 0:
                print(f"epoch {epoch}: loss {total / batches:.4f}", flush=True)
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
