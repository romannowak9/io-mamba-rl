import torch
import torch.nn as nn
import torch.nn.functional as F


class ADNet(nn.Module):
    def __init__(self, num_actions: int, history_length: int = 10):
        super().__init__()

        self.num_actions = num_actions
        self.history_length = history_length
        self.history_dim = num_actions * history_length

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

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
        """
        patch: [B, 3, 112, 112]
        action_history: [B, history_length * num_actions]

        returns:
            action_logits: [B, num_actions]
            confidence_logits: [B, 2]
        """
        x = self.features(patch)
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
    
    def freeze_conv_layers(self):
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    model = ADNet(num_actions=11, history_length=10)

    patch = torch.randn(4, 3, 112, 112)
    history = torch.zeros(4, 110)

    action_logits, confidence_logits = model(patch, history)

    print(action_logits.shape)       # torch.Size([4, 11])
    print(confidence_logits.shape)   # torch.Size([4, 2])