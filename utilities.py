import numpy as np
import torch

def calculate_scalar(x):
    """Calculate mean and std of the input array x."""
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    # frequency-wise mean and std
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return mean, std

def scale(x, mean, std):
    """Scale the input array x using the provided mean and std."""
    epsilon = np.finfo(float).eps
    std = np.maximum(std, epsilon)
    return (x - mean) / std

def calculate_scalar_torch(x):
    """Calculate mean and std of the input tensor x."""
    if x.dim() == 2:
        axis = 0
    elif x.dim() == 3:
        axis = (0, 1)
    else:
        raise ValueError("Input tensor must be 2D or 3D.")
    mean = torch.mean(x, dim=axis)
    std = torch.std(x, dim=axis)
    return mean, std

def scale_torch(x, mean, std):
    """Scale the input tensor x using the provided mean and std."""
    epsilon = torch.finfo(x.dtype).eps
    std = torch.maximum(std, torch.tensor(epsilon, dtype=std.dtype, device=std.device))
    return (x - mean) / std

class Facalloss(torch.nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, gamma=2.0, reduction='mean'):
        super(Facalloss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # test the functions
    x = np.random.rand(100, 1500, 64)  # (n_samples, n_frames, n_mel_bins)
    mean, std = calculate_scalar(x)
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)
    x_scaled = scale(x, mean, std)
    print("Scaled x shape:", x_scaled.shape)