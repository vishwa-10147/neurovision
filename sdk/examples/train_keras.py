"""Train a tiny Keras model with the NeuroVision callback."""

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import torch
from neurovision_callbacks import KerasCallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(128, 3).numpy()
    y = (x[:, 0] > 0).astype("int64")
    model = keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(8, activation="relu", name="hidden"),
            keras.layers.Dense(2, name="output"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=16,
        validation_split=0.25,
        callbacks=[
            KerasCallback(
                url=args.url,
                control=True,
                activation_layers=["hidden"],
                probe_batch=x[:1],
            )
        ],
        verbose=1,
    )


if __name__ == "__main__":
    main()
