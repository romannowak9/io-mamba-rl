import cv2
import numpy as np
import torch
from models.VOT.models.vgg_adnet import VGGMBackbone, ADNet, load_vggm_weights
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS
import torch
from models.VOT.training.datasets import read_groundtruth
from models.VOT.training.datasets import polygon_to_bbox
from pathlib import Path

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
    sample_actions=True,
):
    """
    Runs ADNet action loop on one frame.

    Args:
        frame: OpenCV image, BGR format
        initial_box: [x_center, y_center, w, h]
        gt_box: [x_center, y_center, w, h]
        sample_actions:
            True  -> sample actions for policy-gradient training
            False -> argmax actions for validation/testing

    Returns:
        dict with final_box, actions, log_probs, reward, final_iou
    """
    img_height, img_width = frame.shape[:2]

    box = list(map(float, initial_box))
    history = ActionHistory(action_set, history_length=history_length)

    actions = []
    log_probs = []

    for step in range(max_steps):
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
        # Similar to Model-C idea: reward shorter successful action sequences.
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

if __name__ == "__main__":
    backbone = VGGMBackbone()

    action_set = ORIGINAL_ADNET_ACTIONS
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #load checkpoint
    checkpoint_location = 'checkpoints/adnet_sl_vgg_best_1.pt'
    checkpoint = torch.load(checkpoint_location, map_location=device)
    model = ADNet(
        num_actions=len(action_set),
        history_length=10,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    path_to_file = "data/VOT2015/car1/00000001.jpg"
    groundtruth_path = "data/VOT2015/car1/groundtruth.txt"
    gt_path = Path(groundtruth_path)
    frame = cv2.imread(path_to_file)
    if frame is None:
        raise RuntimeError("Could not read frame")

    groundtruths = read_groundtruth(gt_path)
    
    
    
    current_gt_box = groundtruths[0]

    previous_gt_box = [270, 295, 120, 120]

    print(current_gt_box)

    result = rollout_one_frame(
        model=model,
        frame=frame,
        initial_box=previous_gt_box,
        gt_box=current_gt_box,
        action_set=ORIGINAL_ADNET_ACTIONS,
        device=device,
        sample_actions=False,
    )

    print(result["actions"])
    print(result["final_iou"])
    print(result["reward"])
    final_gt_box = result["final_box"]
    #add bb
    frame_with_bb = frame.copy()

    #transform to corner coords
    x1, y1, x2, y2 = box_to_corners(current_gt_box)
    current_gt_box = (x1, y1, x2, y2)

    x11, y11, x21, y21 = box_to_corners(final_gt_box)
    final_gt_box = (x11, y11, x21, y21)

    x12, y12, x22, y22 = box_to_corners(previous_gt_box)
    previous_gt_box = (x12, y12, x22, y22)

    #RGB0
    cv2.rectangle(frame_with_bb, (int(current_gt_box[0]), int(current_gt_box[1])), (int(current_gt_box[2]), int(current_gt_box[3])), (0, 255, 0), 2)
    cv2.rectangle(frame_with_bb, (int(final_gt_box[0]), int(final_gt_box[1])), (int(final_gt_box[2]), int(final_gt_box[3])), (255, 0, 0), 2)
    cv2.rectangle(frame_with_bb, (int(previous_gt_box[0]), int(previous_gt_box[1])), (int(previous_gt_box[2]), int(previous_gt_box[3])), (0, 0, 255), 2)
    cv2.putText(frame_with_bb, "True GT", (int(current_gt_box[0]), int(current_gt_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame_with_bb, "Predicted GT", (int(final_gt_box[0]), int(final_gt_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame_with_bb, "Initial GT", (int(previous_gt_box[0]), int(previous_gt_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame with BB", frame_with_bb)

    #cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass