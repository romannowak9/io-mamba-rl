from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.VOT.models.vgg_adnet import ADNet, VGGMBackbone, load_vggm_weights
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
from models.VOT.training.dataset_rl import VOTSequenceDataset

from models.VOT.tracking.tracking_on_sequence import rollout_one_frame
from models.VOT.training.supervised import save_checkpoint, EarlyStopping
import time

def make_optimizer_rl(model):
    return torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": 5e-5},
            {"params": model.fc4.parameters(), "lr": 1e-3},
            {"params": model.fc5.parameters(), "lr": 1e-3},
            {"params": model.fc6_action.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
        weight_decay=1e-5,
    )

def rl_collate_fn(batch):
    return batch

def compute_rl_metrics(
    model,
    loader,
    optimizer,
    device,
    action_set,
    max_steps=10,
    reward_type="adnet",
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    totals = {
        "loss": 0.0,
        "mean_reward": 0.0,
        "mean_iou": 0.0,
        "success_rate": 0.0,
        "avg_steps": 0.0,
        "stop_rate": 0.0,
    }

    num_rollouts = 0
    num_batches_with_loss = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    loader_len = len(loader)
    print(f"Number of batches in loader: {loader_len}")

    batch_num = 0
    prev_time = time.time()
    elapsed_time = 0.0

    with context:
        for batch in loader:
            batch_num += 1

            if batch_num % 500 == 0:
                now_time = time.time()
                lapsed_time = now_time - prev_time
                elapsed_time += lapsed_time
                prev_time = now_time

                remaining_batches = loader_len - batch_num
                est_remaining_time = (remaining_batches / 500) * lapsed_time

                print(
                    f"Processing batch {batch_num}/{loader_len}, "
                    f"Est. remaining time in epoch: {est_remaining_time:.2f} s"
                )
                print(f"Elapsed time: {elapsed_time:.2f} s")

            # Safety: allows both batch_size=1 old style and list-collate style.
            if isinstance(batch, dict):
                batch = [batch]

            batch_losses = []

            for sample in batch:
                frames = sample["frames"]
                gt_boxes = sample["gt_boxes"]

                current_box = gt_boxes[0].tolist()
                sequence_losses = []

                for t in range(1, len(frames)):
                    gt_box = gt_boxes[t].tolist()

                    result = rollout_one_frame(
                        model=model,
                        frame=frames[t],
                        initial_box=current_box,
                        gt_box=gt_box,
                        action_set=action_set,
                        device=device,
                        max_steps=max_steps,
                        reward_type=reward_type,
                        sample_actions=is_training,
                    )

                    current_box = result["final_box"]

                    reward = result["reward"]
                    final_iou = result["final_iou"]
                    num_steps = result["num_steps"]
                    stopped = (
                        len(result["actions"]) > 0
                        and result["actions"][-1] == "stop"
                    )

                    if is_training and len(result["log_probs"]) > 0:
                        log_prob_sum = torch.stack(result["log_probs"]).sum()
                        reward_tensor = torch.tensor(
                            reward,
                            dtype=torch.float32,
                            device=device,
                        )
                        loss = -log_prob_sum * reward_tensor
                        sequence_losses.append(loss)

                    totals["mean_reward"] += reward
                    totals["mean_iou"] += final_iou
                    totals["success_rate"] += float(final_iou > 0.7)
                    totals["avg_steps"] += num_steps
                    totals["stop_rate"] += float(stopped)

                    num_rollouts += 1

                if is_training and len(sequence_losses) > 0:
                    sequence_loss = torch.stack(sequence_losses).mean()
                    batch_losses.append(sequence_loss)

            if is_training and len(batch_losses) > 0:
                batch_loss = torch.stack(batch_losses).mean()

                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                totals["loss"] += batch_loss.item()
                num_batches_with_loss += 1

    if num_rollouts == 0:
        raise RuntimeError("No RL rollouts were produced.")

    loss_denominator = max(1, num_batches_with_loss if is_training else loader_len)

    return {
        "loss": totals["loss"] / loss_denominator,
        "mean_reward": totals["mean_reward"] / num_rollouts,
        "mean_iou": totals["mean_iou"] / num_rollouts,
        "success_rate": totals["success_rate"] / num_rollouts,
        "avg_steps": totals["avg_steps"] / num_rollouts,
        "stop_rate": totals["stop_rate"] / num_rollouts,
    }

def print_rl_metrics(epoch, train_metrics, val_metrics):
    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_metrics['loss']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"train_reward={train_metrics['mean_reward']:.3f} | "
        f"val_reward={val_metrics['mean_reward']:.3f} | "
        f"train_iou={train_metrics['mean_iou']:.3f} | "
        f"val_iou={val_metrics['mean_iou']:.3f} | "
        f"train_success={train_metrics['success_rate']:.3f} | "
        f"val_success={val_metrics['success_rate']:.3f} | "
        f"train_steps={train_metrics['avg_steps']:.2f} | "
        f"val_steps={val_metrics['avg_steps']:.2f} | "
        f"train_stop={train_metrics['stop_rate']:.3f} | "
        f"val_stop={val_metrics['stop_rate']:.3f}"
    )

if __name__ == "__main__":

    action_set = ORIGINAL_ADNET_ACTIONS
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("device:", device)

    checkpoint_dir = Path("checkpoints/rl_vot")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    backbone = VGGMBackbone()

    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
        backbone=backbone,
    ).to(device)

    checkpoint_location = "checkpoints/adnet_sl_vgg_best_1.pt"
    checkpoint = torch.load(checkpoint_location, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    
    #load training sequences from file
    training_sequences_path = "train_sequences.txt"
    with open(training_sequences_path, "r") as f:
        training_sequences = {line.strip() for line in f}
    
    rl_train_dataset = VOTSequenceDataset(
        root_dirs=[
            "data/VOT2013",
            "data/VOT2014",
            "data/VOT2015",
            "data/VOT2016",
            "data/OTB/OTB50",
            "data/OTB/OTB100",
            "data/TrackingDataset"
        ],
        clip_length=10,
        stride=2,
        sequence_filter=training_sequences,
    )

    rl_val_dataset = VOTSequenceDataset(
        root_dirs=[
            "data/VOT2013",
            "data/VOT2014",
            "data/VOT2015",
            "data/VOT2016",
            "data/OTB/OTB50",
            "data/OTB/OTB100",
            "data/TrackingDataset"
        ],
        clip_length=10,
        stride=2,
        sequence_filter=training_sequences,
        negative_sequence_filter=True,
    )

    train_loader = DataLoader(
        rl_train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=rl_collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        rl_val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=rl_collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    optimizer = make_optimizer_rl(model)

    num_epochs = 10
    early_stopping = EarlyStopping(
        patience=4,
        min_delta=0.001,
        mode="max",
    )

    for epoch in range(1, num_epochs + 1):
        train_metrics = compute_rl_metrics(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            action_set=action_set,
        )

        val_metrics = compute_rl_metrics(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            action_set=action_set,
        )

        print_rl_metrics(epoch, train_metrics, val_metrics)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            action_set=action_set,
            metrics={"train": train_metrics, "val": val_metrics},
            path=checkpoint_dir / f"rl_training_epoch_{epoch:03d}.pt",
        )

        should_stop, improved = early_stopping.step(val_metrics["mean_iou"])

        if improved:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                action_set=action_set,
                metrics={"train": train_metrics, "val": val_metrics},
                path=checkpoint_dir / "rl_training_best_1.pt",
            )
            print("Saved new best checkpoint.")

        if should_stop:
            print(f"Early stopping at epoch {epoch:03d}.")
            break