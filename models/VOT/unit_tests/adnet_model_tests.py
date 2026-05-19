import torch
from models.VOT.models.adnet import ADNet


def test_adnet_original_action_shape():
    model = ADNet(num_actions=11, history_length=10)

    patch = torch.randn(2, 3, 112, 112)
    history = torch.zeros(2, 110)

    action_logits, confidence_logits = model(patch, history)

    assert action_logits.shape == (2, 11)
    assert confidence_logits.shape == (2, 2)


def test_adnet_drone_action_shape():
    model = ADNet(num_actions=15, history_length=10)

    patch = torch.randn(2, 3, 112, 112)
    history = torch.zeros(2, 150)

    action_logits, confidence_logits = model(patch, history)

    assert action_logits.shape == (2, 15)
    assert confidence_logits.shape == (2, 2)


if __name__ == "__main__":
    test_adnet_original_action_shape()
    test_adnet_drone_action_shape()
    print("All ADNet model tests passed!")