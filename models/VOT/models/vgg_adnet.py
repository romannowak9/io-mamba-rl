import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGMBackbone(nn.Module):
    """
    Siec ADNET z Backbonem z VGG

    Input:
        [B, 3, 112, 112]

    Output:
        [B, 512, 3, 3]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class ADNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        history_length: int = 10,
        backbone: nn.Module | None = None,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.history_length = history_length
        self.history_dim = num_actions * history_length

        self.backbone = backbone if backbone is not None else VGGMBackbone()

        self.fc4 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.fc6_action = nn.Linear(512 + self.history_dim, num_actions)
        self.fc7_confidence = nn.Linear(512 + self.history_dim, 2)

    def forward(self, patch: torch.Tensor, action_history: torch.Tensor):
        x = self.backbone(patch)
        x = torch.flatten(x, start_dim=1)

        x = self.fc4(x)
        x = self.fc5(x)

        x = torch.cat([x, action_history], dim=1)

        action_logits = self.fc6_action(x)
        confidence_logits = self.fc7_confidence(x)

        return action_logits, confidence_logits

    def action_probabilities(self, patch, action_history):
        action_logits, _ = self.forward(patch, action_history)
        return F.softmax(action_logits, dim=1)

    def confidence_probabilities(self, patch, action_history):
        _, confidence_logits = self.forward(patch, action_history)
        return F.softmax(confidence_logits, dim=1)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def load_vggm_weights(backbone: VGGMBackbone, weight_path: str, strict: bool = False):
    """
    Loads compatible VGG-M pretrained weights into the ADNet backbone.

    Because pretrained VGG-M checkpoints often contain classifier layers too,
    we load only matching convolutional weights.
    """
    checkpoint = torch.load(weight_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    backbone_state = backbone.state_dict()
    filtered = {}

    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "")

        # Common possibilities:
        # features.0.weight, features.0.bias, etc.
        if clean_key in backbone_state and backbone_state[clean_key].shape == value.shape:
            filtered[clean_key] = value

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)

    print(f"Loaded {len(filtered)} VGG-M tensors into backbone.")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    if strict and len(filtered) == 0:
        raise RuntimeError("No compatible VGG-M weights were loaded.")

    return backbone


if __name__ == "__main__":
    backbone = VGGMBackbone()
    load_vggm_weights(backbone, "models/pretrained/vggm.pth")

    model = ADNet(
        num_actions=11,
        history_length=10,
        backbone=backbone,
    )

    patch = torch.randn(4, 3, 112, 112)
    history = torch.zeros(4, 110)

    action_logits, confidence_logits = model(patch, history)

    print(action_logits.shape)
    print(confidence_logits.shape)