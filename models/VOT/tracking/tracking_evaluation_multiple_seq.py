import csv
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
from models.VOT.tracking.tracking_on_sequence import find_frames, rollout_one_frame, draw_box


def load_sequence_list(txt_path):
    with open(txt_path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def save_results_csv(results, output_csv):
    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "sequence",
            "mean_iou",
            "success_rate",
            "avg_steps",
            "mean_reward",
            "final_iou",
            "frames_evaluated",
        ])

        for r in results:
            writer.writerow([
                r["sequence"],
                f"{r['mean_iou']:.6f}",
                f"{r['success_rate']:.6f}",
                f"{r['avg_steps']:.6f}",
                f"{r['mean_reward']:.6f}",
                f"{r['final_iou']:.6f}",
                r["frames_evaluated"],
            ])


def evaluate_sequence(
    model,
    sequence_dir,
    action_set,
    device,
    start_frame=0,
    max_frames=None,
    display=True,
    save_video_path=None,
    history_length=10
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

    history = ActionHistory(action_set, history_length=history_length)

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
            history=history,
            sample_actions=False,
        )

        time_2 = time.time()

        duration = time_2 - time_1
        total_eval_duration += duration

        predicted_box = result["final_box"]
        current_box = predicted_box
        history = result["history"]

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
        "final_iou": result["final_iou"],
        "frames_evaluated": len(ious),
        "ious": ious,
        "rewards": rewards,
        "steps": steps,
    }


def evaluate_multiple_sequences(
    model,
    sequence_list_file,
    dataset_root="data",
    action_set=None,
    device="cuda",
    max_frames=None,
    output_csv="sequence_results.csv",
):

    sequences = load_sequence_list(sequence_list_file)

    print(f"Loaded {len(sequences)} sequences")

    all_results = []

    mean_ious = []
    success_rates = []
    rewards = []
    avg_steps = []

    final_success_count = 0

    for seq_idx, sequence_name in enumerate(sequences):

        sequence_dir = Path(dataset_root) / sequence_name

        print(
            f"\n[{seq_idx+1}/{len(sequences)}] "
            f"{sequence_name}"
        )

        try:

            metrics = evaluate_sequence(
                model=model,
                sequence_dir=sequence_dir,
                action_set=action_set,
                device=device,
                start_frame=0,
                max_frames=max_frames,
                display=False,
                save_video_path=None,
            )

            result = {
                "sequence": sequence_name,
                **metrics,
            }

            all_results.append(result)

            mean_ious.append(metrics["mean_iou"])
            success_rates.append(metrics["success_rate"])
            rewards.append(metrics["mean_reward"])
            avg_steps.append(metrics["avg_steps"])

            if metrics["final_iou"] >= 0.7:
                final_success_count += 1

        except Exception as e:

            print(
                f"FAILED: {sequence_name}"
            )

            print(e)

    num_sequences = len(all_results)

    summary = {
        "mean_iou":
            float(np.mean(mean_ious))
            if mean_ious else 0.0,

        "mean_success_rate":
            float(np.mean(success_rates))
            if success_rates else 0.0,

        "mean_reward":
            float(np.mean(rewards))
            if rewards else 0.0,

        "mean_steps":
            float(np.mean(avg_steps))
            if avg_steps else 0.0,

        "num_sequences":
            num_sequences,

        "final_success_count":
            final_success_count,

        "final_success_ratio":
            (
                final_success_count / num_sequences
                if num_sequences > 0
                else 0.0
            ),
    }

    save_results_csv(
        all_results,
        output_csv,
    )

    print("\n")
    print("=" * 60)
    print("GLOBAL RESULTS")
    print("=" * 60)

    print(
        f"Sequences evaluated: "
        f"{summary['num_sequences']}"
    )

    print(
        f"Mean IoU: "
        f"{summary['mean_iou']:.4f}"
    )

    print(
        f"Mean Success Rate: "
        f"{summary['mean_success_rate']:.4f}"
    )

    print(
        f"Mean Reward: "
        f"{summary['mean_reward']:.4f}"
    )

    print(
        f"Mean Steps: "
        f"{summary['mean_steps']:.2f}"
    )

    print(
        f"Final IoU >= 0.7: "
        f"{summary['final_success_count']}"
        f"/{summary['num_sequences']}"
        f" ({summary['final_success_ratio']:.4f})"
    )

    print("\n")

    all_results.sort(
        key=lambda x: x["final_iou"],
        reverse=True,
    )

    print("=" * 60)
    print("TOP 10 SEQUENCES")
    print("=" * 60)

    for r in all_results[:10]:

        print(
            f"{r['sequence']:<45}"
            f"{r['final_iou']:.4f}"
        )

    print("\n")

    print("=" * 60)
    print("BOTTOM 10 SEQUENCES")
    print("=" * 60)

    for r in all_results[-10:]:

        print(
            f"{r['sequence']:<45}"
            f"{r['final_iou']:.4f}"
        )

    return summary, all_results


if __name__ == "__main__":
    action_set = ORIGINAL_ADNET_ACTIONS
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    backbone = VGGMBackbone()

    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
        backbone=backbone,
    ).to(device)

    #checkpoint_location = "checkpoints/adnet_sl_vgg_best_1.pt"
    checkpoint_location = "checkpoints/rl_vot/rl_training_best_afo_1.pt"
    checkpoint = torch.load(checkpoint_location, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary, results = evaluate_multiple_sequences(
        model=model,
        sequence_list_file="train_sequences_afo.txt",
        dataset_root="data",
        action_set=action_set,
        device=device,
        max_frames=200,
        output_csv="vot_train_sequences_results.csv",
    )