"""Demo: reinforcement learning, a policy learning to balance a pole.

CartPole: push a cart left or right to keep a hinged pole upright. The
physics are the classic ones (Barto, Sutton & Anderson, 1983), written out
here so nothing needs installing. The episode ends when the pole tilts more
than 12° or the cart leaves the track; 500 steps is the maximum score. A
small policy network learns with REINFORCE: actions that led to long
episodes become more likely. Live Runs shows the policy's graph, the
average and best episode length per iteration, and the pole angle over the
latest episode.

    python sdk/examples/train_cartpole.py --url http://127.0.0.1:8001
"""

import math
import time

import torch
from demo_common import arguments, nv
from torch import nn

MAX_STEPS = 500


class CartPole:
    """Classic cart-pole dynamics, Euler integration at 50 Hz."""

    gravity, cart_mass, pole_mass, half_length, force, dt = (
        9.8,
        1.0,
        0.1,
        0.5,
        10.0,
        0.02,
    )

    def __init__(self, generator: torch.Generator) -> None:
        self.generator = generator

    def reset(self) -> list[float]:
        self.state = (torch.rand(4, generator=self.generator) * 0.1 - 0.05).tolist()
        return self.state

    def step(self, action: int) -> tuple[list[float], bool]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force if action == 1 else -self.force
        total_mass = self.cart_mass + self.pole_mass
        pole_moment = self.pole_mass * self.half_length
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + pole_moment * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4 / 3 - self.pole_mass * cos**2 / total_mass)
        )
        x_acc = temp - pole_moment * theta_acc * cos / total_mass
        x, x_dot = x + self.dt * x_dot, x_dot + self.dt * x_acc
        theta, theta_dot = theta + self.dt * theta_dot, theta_dot + self.dt * theta_acc
        self.state = [x, x_dot, theta, theta_dot]
        failed = abs(x) > 2.4 or abs(theta) > 12 * math.pi / 180
        return self.state, failed


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # Action logits: push left, push right.


def main() -> None:
    args = arguments(__doc__.splitlines()[0], epochs=150, lr=0.01)
    torch.manual_seed(0)
    env = CartPole(torch.Generator().manual_seed(0))
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    probe = torch.tensor([[0.0, 0.0, 0.05, 0.0], [0.0, 0.0, -0.05, 0.0]] * 8)
    monitor = nv.HealthMonitor(policy, optimizer=optimizer, probe_batch=probe)
    episodes_per_iteration = 10
    with nv.start_run(
        args.name or "REINFORCE on CartPole",
        total_epochs=args.epochs,
        url=args.url,
        model=None if args.no_graph else policy,
        example_input=probe[:1],
        control=True,
        optimizer=optimizer,
    ) as run:
        for iteration in range(1, args.epochs + 1):
            if not run.poll():
                break
            log_probs, returns, lengths = [], [], []
            angles: list[float] = []
            for _ in range(episodes_per_iteration):
                state, rewards, angles = env.reset(), [], []
                for _ in range(MAX_STEPS):
                    logits = policy(torch.tensor(state))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_probs.append(dist.log_prob(action))
                    state, failed = env.step(int(action))
                    rewards.append(1.0)
                    angles.append(math.degrees(state[2]))
                    if failed:
                        break
                # Return-to-go per step, discounted.
                g, episode_returns = 0.0, []
                for r in reversed(rewards):
                    g = r + 0.99 * g
                    episode_returns.append(g)
                returns.extend(reversed(episode_returns))
                lengths.append(len(rewards))
            returns_t = torch.tensor(returns)
            advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            loss = -(torch.stack(log_probs) * advantages).sum() / episodes_per_iteration
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_length = sum(lengths) / len(lengths)
            run.log(
                epoch=iteration,
                metrics={
                    "mean episode length": mean_length,
                    "best episode length": max(lengths),
                    # Not "loss": a policy-gradient loss is not expected to fall steadily,
                    # so the loss spike/plateau rules do not apply to it.
                    "policy_loss": loss.item(),
                },
                health=monitor.collect(),
                visuals={
                    "pole angle in the latest episode (degrees)": nv.scatter(
                        list(range(len(angles))),
                        angles,
                        x_label="time step",
                        y_label="angle (°)",
                    )
                },
            )
            time.sleep(args.delay)  # --delay: slow motion, to watch live.
            if iteration == 1 or iteration % 15 == 0:
                print(
                    f"iteration {iteration}: mean length {mean_length:.0f}, best {max(lengths)}",
                    flush=True,
                )
    print(f"done; sent {run.sent} steps")


if __name__ == "__main__":
    main()
