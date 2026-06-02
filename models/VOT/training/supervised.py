from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.VOT.models.adnet import ADNet
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
from models.VOT.training.datasets import VOTSupervisedDataset, list_vot_sequences


def make_optimizer(model):
    return torch.optim.SGD(
        [
            {"params": model.features.parameters(), "lr": 2e-4},
            {"params": model.fc4.parameters(), "lr": 2e-3},
            {"params": model.fc5.parameters(), "lr": 2e-3},
            {"params": model.fc6_action.parameters(), "lr": 2e-3},
            {"params": model.fc7_confidence.parameters(), "lr": 2e-3},
        ],
        momentum=0.9,
        weight_decay=1e-5,
    )


def compute_metrics(model, loader, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    action_criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()

    totals = {
        "loss": 0.0,
        "action_loss": 0.0,
        "class_loss": 0.0,
        "action_acc": 0.0,
        "class_acc": 0.0,
        "positive_ratio": 0.0,
    }

    num_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            patches = batch["patch"].to(device, non_blocking=True)
            histories = batch["history"].to(device, non_blocking=True)
            action_labels = batch["action_label"].to(device, non_blocking=True)
            class_labels = batch["class_label"].to(device, non_blocking=True)

            action_logits, confidence_logits = model(patches, histories)

            action_loss = action_criterion(action_logits, action_labels)
            class_loss = class_criterion(confidence_logits, class_labels)
            loss = action_loss + class_loss

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            action_pred = action_logits.argmax(dim=1)
            class_pred = confidence_logits.argmax(dim=1)

            totals["loss"] += loss.item()
            totals["action_loss"] += action_loss.item()
            totals["class_loss"] += class_loss.item()
            totals["action_acc"] += (action_pred == action_labels).float().mean().item()
            totals["class_acc"] += (class_pred == class_labels).float().mean().item()
            totals["positive_ratio"] += class_labels.float().mean().item()

            num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def save_checkpoint(model, optimizer, epoch, action_set, path, metrics=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "action_set": action_set,
            "metrics": metrics or {},
        },
        path,
    )


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False, True

        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False, True

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return should_stop, False


def print_metrics(epoch, train_metrics, val_metrics):
    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_metrics['loss']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"train_action_acc={train_metrics['action_acc']:.3f} | "
        f"val_action_acc={val_metrics['action_acc']:.3f} | "
        f"train_class_acc={train_metrics['class_acc']:.3f} | "
        f"val_class_acc={val_metrics['class_acc']:.3f} | "
        f"train_pos={train_metrics['positive_ratio']:.3f} | "
        f"val_pos={val_metrics['positive_ratio']:.3f}"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    action_set = ORIGINAL_ADNET_ACTIONS

    root_dirs = [
        "data/VOT2013",
        "data/VOT2014",
        "data/VOT2015",
        "data/VOT2016",
        "data/OTB/OTB50",
        "data/OTB/OTB100"
    ]

    all_sequences = list_vot_sequences(root_dirs)

    random.seed(42)
    random.shuffle(all_sequences)

    val_fraction = 0.2
    val_count = max(1, int(len(all_sequences) * val_fraction))

    val_sequences = set(all_sequences[:val_count])
    train_sequences = set(all_sequences[val_count:])

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")

    train_dataset = VOTSupervisedDataset(
        root_dirs=root_dirs,
        action_set=action_set,
        samples_per_frame=10,
        input_size=(112, 112),
        history_length=10,
        sequence_filter=train_sequences,
        target_positive_ratio=0.3,
    )

    val_dataset = VOTSupervisedDataset(
        root_dirs=root_dirs,
        action_set=action_set,
        samples_per_frame=3,
        input_size=(112, 112),
        history_length=10,
        sequence_filter=val_sequences,
        target_positive_ratio=0.3,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
    ).to(device)

    optimizer = make_optimizer(model)

    num_epochs = 30
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001,
        mode="max",
    )

    for epoch in range(1, num_epochs + 1):
        train_metrics = compute_metrics(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = compute_metrics(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
        )

        print_metrics(epoch, train_metrics, val_metrics)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            action_set=action_set,
            metrics={"train": train_metrics, "val": val_metrics},
            path=checkpoint_dir / f"adnet_sl_epoch_{epoch:03d}.pt",
        )

        should_stop, improved = early_stopping.step(val_metrics["action_acc"])

        if improved:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                action_set=action_set,
                metrics={"train": train_metrics, "val": val_metrics},
                path=checkpoint_dir / "adnet_sl_best.pt",
            )
            print("Saved new best checkpoint.")

        if should_stop:
            print(f"Early stopping at epoch {epoch:03d}.")
            break


if __name__ == "__main__":
    main()