import torch


class AddGaussianNoise:

    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"