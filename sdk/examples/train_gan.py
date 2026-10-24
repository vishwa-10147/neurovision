"""Demo: a GAN learning to draw handwritten digits, live.

Two networks compete: the generator turns random noise into 8×8 images and
the discriminator tries to tell them from real digits (scikit-learn's 1,797
images, no download). Live Runs shows the generator's graph, both losses,
how confident the discriminator is on real and fake images, and a grid of
16 generated digits from the same fixed noise every epoch, so you can watch
them take shape.

    pip install scikit-learn
    python sdk/examples/train_gan.py --url http://127.0.0.1:8001
"""

import time

import torch
from demo_common import arguments, device_for, nv
from torch import nn
from train_digits_cnn import digits

NOISE = 16


def generator() -> nn.Module:
    return nn.Sequential(
        nn.Linear(NOISE, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 64), nn.Sigmoid(),
    )  # fmt: skip


def discriminator() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
        nn.Linear(128, 1),
    )  # fmt: skip


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=150, lr=0.0005)
    device = device_for(args.device)
    torch.manual_seed(0)
    x_train, _, _, _ = digits()
    real_images = x_train.flatten(1).to(device)
    g, d = generator().to(device), discriminator().to(device)
    g_opt = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(16, NOISE, device=device)
    monitor = nv.HealthMonitor(g, optimizer=g_opt, probe_batch=fixed_noise)
    step = 0
    with (
        nv.start_run(
            args.name or "GAN drawing digits",
            total_epochs=args.epochs,
            url=args.url,
            model=None if args.no_graph else g,  # The generator's graph.
            example_input=fixed_noise[:1],
            control=True,
            optimizer=g_opt,
        ) as run
    ):
        for epoch in range(1, args.epochs + 1):
            sums = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
            batches = 0
            for index in torch.randperm(len(real_images), device=device).split(
                args.batch
            ):
                if not run.poll():
                    break
                real = real_images[index]
                fake = g(torch.randn(len(real), NOISE, device=device))
                # Discriminator: real → 1 (smoothed to 0.9), fake → 0.
                d_opt.zero_grad()
                d_real, d_fake = d(real), d(fake.detach())
                d_loss = bce(d_real, torch.full_like(d_real, 0.9)) + bce(
                    d_fake, torch.zeros_like(d_fake)
                )
                d_loss.backward()
                d_opt.step()
                # Generator: make the discriminator say 1 for fakes.
                g_opt.zero_grad()
                g_loss = bce(d(fake), torch.ones_like(d_fake))
                g_loss.backward()
                g_opt.step()
                step += 1
                batches += 1
                sums["d_loss"] += d_loss.item()
                sums["g_loss"] += g_loss.item()
                sums["d_real"] += torch.sigmoid(d_real).mean().item()
                sums["d_fake"] += torch.sigmoid(d_fake).mean().item()
            if batches == 0:
                break
            g.eval()
            with torch.no_grad():
                samples = g(fixed_noise).reshape(16, 8, 8)
            g.train()
            metrics = {k: v / batches for k, v in sums.items()}
            run.log(
                epoch=epoch,
                step=step,
                metrics=metrics,
                health=monitor.collect(),
                samples=len(real_images),
                visuals={
                    "generated digits (fixed noise)": nv.matrix(
                        nv.image_grid(samples, columns=4), scale="sequential"
                    ),
                },
            )
            if run.should_stop:
                break
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if epoch == 1 or epoch % 25 == 0:
                shown = ", ".join(f"{k} {v:.3f}" for k, v in metrics.items())
                print(f"epoch {epoch}: {shown}", flush=True)
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
