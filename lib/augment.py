import torch

def augment(image_tensor: torch.Tensor) -> list[torch.Tensor]:
    return [
        image_tensor,
        image_tensor.flip(-1),
        image_tensor.flip(-2),
        image_tensor.rot90(dims=(-2, -1)),
        image_tensor.rot90(dims=(-1, -2)),
    ]
