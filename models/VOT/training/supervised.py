from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.VOT.models.adnet import ADNet
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
from models.VOT.training.datasets import VOTSupervisedDataset


def make_optimizer(model):
    return torch.optim.SGD(
        [
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.fc4.parameters(), "lr": 1e-3},
            {"params": model.fc5.parameters(), "lr": 1e-3},
            {"params": model.fc6_action.parameters(), "lr": 1e-3},
            {"params": model.fc7_confidence.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()

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

    for batch in loader:
        patches = batch["patch"].to(device, non_blocking=True)
        histories = batch["history"].to(device, non_blocking=True)
        action_labels = batch["action_label"].to(device, non_blocking=True)
        class_labels = batch["class_label"].to(device, non_blocking=True)

        action_logits, confidence_logits = model(patches, histories)

        action_loss = action_criterion(action_logits, action_labels)
        class_loss = class_criterion(confidence_logits, class_labels)
        loss = action_loss + class_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            action_pred = action_logits.argmax(dim=1)
            class_pred = confidence_logits.argmax(dim=1)

            action_acc = (action_pred == action_labels).float().mean()
            class_acc = (class_pred == class_labels).float().mean()
            positive_ratio = class_labels.float().mean()

        totals["loss"] += loss.item()
        totals["action_loss"] += action_loss.item()
        totals["class_loss"] += class_loss.item()
        totals["action_acc"] += action_acc.item()
        totals["class_acc"] += class_acc.item()
        totals["positive_ratio"] += positive_ratio.item()

        num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def save_checkpoint(model, optimizer, epoch, action_set, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "action_set": action_set,
        },
        path,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    action_set = ORIGINAL_ADNET_ACTIONS

    dataset = VOTSupervisedDataset(
        root_dirs=[
            "data/VOT2013",
            "data/VOT2014",
            "data/VOT2015",
        ],
        action_set=action_set,
        samples_per_frame=10,
        input_size=(112, 112),
        history_length=10,
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
    ).to(device)

    optimizer = make_optimizer(model)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={metrics['loss']:.4f} | "
            f"action_loss={metrics['action_loss']:.4f} | "
            f"class_loss={metrics['class_loss']:.4f} | "
            f"action_acc={metrics['action_acc']:.3f} | "
            f"class_acc={metrics['class_acc']:.3f} | "
            f"positive_ratio={metrics['positive_ratio']:.3f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            action_set=action_set,
            path=checkpoint_dir / f"adnet_sl_epoch_{epoch:03d}.pt",
        )


if __name__ == "__main__":
    main()