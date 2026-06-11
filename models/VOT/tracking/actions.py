from dataclasses import dataclass
from typing import Sequence

from models.VOT.utils.bbox import clip_box

import numpy as np


@dataclass(frozen=True)
class ActionConfig:
    alpha: float = 0.05
    scale_up: float = 1.25
    scale_down: float = 0.8


ORIGINAL_ADNET_ACTIONS = [
    "left",
    "right",
    "up",
    "down",
    "large-left",
    "large-right",
    "large-up",
    "large-down",
    "scale-up",
    "scale-down",
    "stop",
]


DRONE_ADNET_ACTIONS = [
    "left",
    "right",
    "up",
    "down",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
    "large-left",
    "large-right",
    "large-up",
    "large-down",
    "scale-up",
    "scale-down",
    "stop",
]


def action_to_index(action: str, action_set: Sequence[str]) -> int:
    return action_set.index(action)


def index_to_action(index: int, action_set: Sequence[str]) -> str:
    return action_set[index]


def apply_action(box, action: str, config: ActionConfig = ActionConfig()):
    """
    box format: [x_center, y_center, width, height]
    """
    x, y, w, h = map(float, box)

    dx = config.alpha * w
    dy = config.alpha * h

    if action == "left":
        x -= dx
    elif action == "right":
        x += dx
    elif action == "up":
        y -= dy
    elif action == "down":
        y += dy

    elif action == "large-left":
        x -= 2 * dx
    elif action == "large-right":
        x += 2 * dx
    elif action == "large-up":
        y -= 2 * dy
    elif action == "large-down":
        y += 2 * dy

    elif action == "up-left":
        x -= dx
        y -= dy
    elif action == "up-right":
        x += dx
        y -= dy
    elif action == "down-left":
        x -= dx
        y += dy
    elif action == "down-right":
        x += dx
        y += dy

    elif action == "scale-up":
        w *= config.scale_up
        h *= config.scale_up
    elif action == "scale-down":
        w *= config.scale_down
        h *= config.scale_down

    elif action == "stop":
        pass
    else:
        raise ValueError(f"Unknown action: {action}")

    return [x, y, w, h]


def transition_box(
    box,
    action: str,
    img_width: int,
    img_height: int,
    config: ActionConfig = ActionConfig(),
):
    next_box = apply_action(box, action, config)
    next_box = clip_box(next_box, img_width, img_height)
    return next_box


class ActionHistory:
    def __init__(self, action_set, history_length=10):
        self.action_set = action_set
        self.history_length = history_length
        self.num_actions = len(action_set)
        self.reset()

    def reset(self):
        self.history = []

    def push(self, action: str):
        if action not in self.action_set:
            raise ValueError(f"Unknown action: {action}")

        self.history.insert(0, action)
        self.history = self.history[: self.history_length]

    def as_vector(self):
        vec = np.zeros(self.history_length * self.num_actions, dtype=np.float32)

        for t, action in enumerate(self.history):
            action_idx = action_to_index(action, self.action_set)
            vec[t * self.num_actions + action_idx] = 1.0

        return vec

