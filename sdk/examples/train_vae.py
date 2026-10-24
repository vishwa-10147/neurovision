"""Demo: a variational autoencoder organizing digits in a 2D latent space.

The encoder maps each 8×8 digit (scikit-learn's 1,797 images, no download)
to a point in 2D; the decoder rebuilds the image from that point. Live Runs
shows the graph, the reconstruction and KL losses, the latent space of all
held-out digits colored by their true class (the model never sees the
labels; the clusters form on their own), and originals next to their
reconstructions.

    pip install scikit-learn
    python sdk/examples/train_vae.py --url http://127.0.0.1:8001
"""

import time

import torch
from demo_common import arguments, device_for, nv
from torch import nn
from torch.nn import functional as F
from train_digits_cnn import digits

LATENT = 2


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.mean = nn.Linear(64, LATENT)
        self.log_var = nn.Linear(64, LATENT)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mean(h), self.log_var(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encode(x)
        # Sample while training; use the mean otherwise (and in the export).
        z = (
            mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
            if self.training
            else mean
        )
        return self.decoder(z)


def losses(model: VAE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_var = model.encode(x)
    z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
    reconstruction = F.binary_cross_entropy(model.decoder(z), x, reduction="sum") / len(
        x
    )
    kl = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp()) / len(x)
    return reconstruction, kl


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=80, lr=0.002)
    device = device_for(args.device)
    torch.manual_seed(0)
    x_train, _, x_val, y_val = digits()
    x_train, x_val = x_train.flatten(1).to(device), x_val.flatten(1).to(device)
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    monitor = nv.HealthMonitor(model, optimizer=optimizer, probe_batch=x_val[:64])
    step = 0
    with nv.start_run(
        args.name or "VAE latent space (digits)",
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else model,
        example_input=x_val[:1],
        control=True,
        optimizer=optimizer,
    ) as run:
        for epoch in range(1, args.epochs + 1):
            sums = {"reconstruction": 0.0, "kl": 0.0}
            batches = 0
            for index in torch.randperm(len(x_train), device=device).split(args.batch):
                if not run.poll():
                    break
                optimizer.zero_grad()
                reconstruction, kl = losses(model, x_train[index])
                (reconstruction + kl).backward()
                optimizer.step()
                step += 1
                batches += 1
                sums["reconstruction"] += reconstruction.item()
                sums["kl"] += kl.item()
            if batches == 0:
                break
            model.eval()
            with torch.no_grad():
                val_reconstruction, val_kl = losses(model, x_val)
                mean, _ = model.encode(x_val)
                rebuilt = model(x_val[:8])
            model.train()
            metrics = {
                "loss": (sums["reconstruction"] + sums["kl"]) / batches,
                "reconstruction": sums["reconstruction"] / batches,
                "kl": sums["kl"] / batches,
                "val_loss": (val_reconstruction + val_kl).item(),
            }
            pairs = torch.cat([x_val[:8], rebuilt]).reshape(16, 8, 8)
            run.log(
                epoch=epoch,
                step=step,
                metrics=metrics,
                health=monitor.collect(),
                samples=len(x_train),
                visuals={
                    "latent space (held-out digits, colored by class)": nv.scatter(
                        mean[:, 0], mean[:, 1], labels=y_val, x_label="z1", y_label="z2"
                    ),
                    "originals (top two rows) and their reconstructions": nv.matrix(
                        nv.image_grid(pairs, columns=4), scale="sequential"
                    ),
                },
            )
            if run.should_stop:
                break
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch {epoch}: "
                    + ", ".join(f"{k} {v:.2f}" for k, v in metrics.items()),
                    flush=True,
                )
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
