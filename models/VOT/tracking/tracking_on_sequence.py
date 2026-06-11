import cv2
import numpy as np
import torch
from pathlib import Path
import time

from models.VOT.models.vgg_adnet import VGGMBackbone, ADNet
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
from models.VOT.training.datasets import read_groundtruth
from models.VOT.tracking.actions import (
    ActionHistory,
    index_to_action,
    transition_box,
)
from models.VOT.utils.bbox import iou
from models.VOT.utils.crops import crop_patch, resize_patch


def patch_to_tensor(patch, device):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
    return patch.unsqueeze(0).to(device)


def crop_box_as_tensor(frame, box, input_size, device):
    patch = crop_patch(frame, box)

    if patch is None or patch.size == 0:
        patch = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    else:
        patch = resize_patch(patch, input_size)

    return patch_to_tensor(patch, device)


def rollout_one_frame(
    model,
    frame,
    initial_box,
    gt_box,
    action_set,
    device,
    input_size=(112, 112),
    history_length=10,
    max_steps=10,
    reward_type="adnet",
    sample_actions=False,
):
    img_height, img_width = frame.shape[:2]

    box = list(map(float, initial_box))
    history = ActionHistory(action_set, history_length=history_length)

    actions = []
    log_probs = []

    with torch.no_grad() if not sample_actions else torch.enable_grad():
        for _ in range(max_steps):
            patch_tensor = crop_box_as_tensor(
                frame=frame,
                box=box,
                input_size=input_size,
                device=device,
            )

            history_tensor = torch.from_numpy(history.as_vector()).float()
            history_tensor = history_tensor.unsqueeze(0).to(device)

            action_logits, _ = model(patch_tensor, history_tensor)
            action_probs = torch.softmax(action_logits, dim=1)

            if sample_actions:
                dist = torch.distributions.Categorical(probs=action_probs)
                action_idx_tensor = dist.sample()
                log_prob = dist.log_prob(action_idx_tensor)

                action_idx = int(action_idx_tensor.item())
                log_probs.append(log_prob)
            else:
                action_idx = int(action_probs.argmax(dim=1).item())

            action = index_to_action(action_idx, action_set)
            actions.append(action)

            if action == "stop":
                break

            box = transition_box(
                box=box,
                action=action,
                img_width=img_width,
                img_height=img_height,
            )

            history.push(action)

    final_iou = iou(box, gt_box)

    if reward_type == "adnet":
        reward = 1.0 if final_iou > 0.7 else -1.0
    elif reward_type == "drone":
        reward = (max_steps - len(actions)) * final_iou if final_iou > 0.7 else -1.0
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    return {
        "final_box": box,
        "actions": actions,
        "log_probs": log_probs,
        "reward": reward,
        "final_iou": final_iou,
        "num_steps": len(actions),
    }


def box_to_corners(box):
    x_center, y_center, width, height = box

    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))

    return x1, y1, x2, y2


def find_frames(sequence_dir):
    sequence_dir = Path(sequence_dir)

    frames = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        frames.extend(sequence_dir.glob(ext))

    return sorted(frames)


def draw_box(frame, box, color, label):
    x1, y1, x2, y2 = box_to_corners(box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def evaluate_sequence(
    model,
    sequence_dir,
    action_set,
    device,
    start_frame=0,
    max_frames=None,
    display=True,
    save_video_path=None,
):
    sequence_dir = Path(sequence_dir)

    frames = find_frames(sequence_dir)
    gt_boxes = read_groundtruth(sequence_dir / "groundtruth.txt")

    total_eval_duration = 0

    n = min(len(frames), len(gt_boxes))

    if max_frames is not None:
        n = min(n, start_frame + max_frames)

    if n <= start_frame + 1:
        raise RuntimeError("Not enough frames to evaluate sequence.")

    first_frame = cv2.imread(str(frames[start_frame]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {frames[start_frame]}")

    video_writer = None

    if save_video_path is not None:
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(save_video_path), fourcc, 20.0, (w, h))

    current_box = gt_boxes[start_frame]

    ious = []
    rewards = []
    steps = []

    total_loop_time = 0
    prev_loop_time = time.time()

    for frame_idx in range(start_frame + 1, n):

        frame = cv2.imread(str(frames[frame_idx]))
        if frame is None:
            print(f"Could not read frame: {frames[frame_idx]}")
            continue

        gt_box = gt_boxes[frame_idx]
        initial_box = current_box

        #record time
        time_1 = time.time()

        result = rollout_one_frame(
            model=model,
            frame=frame,
            initial_box=initial_box,
            gt_box=gt_box,
            action_set=action_set,
            device=device,
            sample_actions=False,
        )

        time_2 = time.time()

        duration = time_2 - time_1
        total_eval_duration += duration

        predicted_box = result["final_box"]
        current_box = predicted_box

        ious.append(result["final_iou"])
        rewards.append(result["reward"])
        steps.append(result["num_steps"])

        vis = frame.copy()

        # Green: true GT
        draw_box(vis, gt_box, (0, 255, 0), "GT")

        # Blue: predicted final box
        draw_box(vis, predicted_box, (255, 0, 0), "Pred")

        # Red: initial box before rollout
        draw_box(vis, initial_box, (0, 0, 255), "Init")

        cv2.putText(
            vis,
            f"frame={frame_idx} IoU={result['final_iou']:.3f} "
            f"reward={result['reward']:.1f} steps={result['num_steps']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "actions: " + ",".join(result["actions"][:8]),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if video_writer is not None:
            video_writer.write(vis)

        if display:
            cv2.imshow("ADNet sequence evaluation", vis)
            key = cv2.waitKey(30)

            if key == 27:  # ESC
                break
        
        loop_time = time.time()
        total_loop_time += loop_time - prev_loop_time
        prev_loop_time = loop_time
        

    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()

    mean_iou = float(np.mean(ious)) if ious else 0.0
    success_rate = float(np.mean([v > 0.7 for v in ious])) if ious else 0.0
    avg_steps = float(np.mean(steps)) if steps else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    print("Sequence evaluation finished.")
    print(f"Frames evaluated: {len(ious)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Success rate IoU>0.7: {success_rate:.4f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Mean reward: {mean_reward:.4f}")

    print(f"Final frame IOU: {result['final_iou']:.4f}")
    cv2.imshow("Final frame", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Total evaluation duration: {total_eval_duration:.2f} seconds")
    print(f"Average evaluation duration per frame: {total_eval_duration / len(ious):.4f} seconds" if ious else "No frames evaluated")
    print(f"Total loop time (with display and read): {total_loop_time:.2f} seconds")
    print(f"Average loop time per frame (with display and read): {total_loop_time / len(ious):.4f} seconds" if ious else "No frames evaluated")

    return {
        "mean_iou": mean_iou,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "mean_reward": mean_reward,
        "ious": ious,
        "rewards": rewards,
        "steps": steps,
    }


if __name__ == "__main__":
    action_set = ORIGINAL_ADNET_ACTIONS
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    backbone = VGGMBackbone()

    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
        backbone=backbone,
    ).to(device)

    checkpoint_location = "checkpoints/adnet_sl_vgg_best_1.pt"
    checkpoint = torch.load(checkpoint_location, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluate_sequence(
        model=model,
        sequence_dir="data/VOT2015/car1",
        action_set=action_set,
        device=device,
        start_frame=0,
        max_frames=200,
        display=True,
        save_video_path="debug_adnet_car1.mp4",
    )