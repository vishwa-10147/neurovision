"""Train a tiny Hugging Face model built from a config, without downloads."""

import argparse

import torch
from neurovision_callbacks import HuggingFaceCallback
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class Rows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int = 64) -> None:
        generator = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(0, 50, (size, 8), generator=generator)
        self.attention_mask = torch.ones(size, 8, dtype=torch.long)
        self.labels = torch.randint(0, 2, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
    )
    training_args = TrainingArguments(
        output_dir="./hf-neurovision-output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
    )
    Trainer(
        model=model,
        args=training_args,
        train_dataset=Rows(),
        eval_dataset=Rows(32),
        callbacks=[HuggingFaceCallback(url=args.url)],
    ).train()


if __name__ == "__main__":
    main()
