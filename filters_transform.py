import torch
import numpy as np


class ColorFilter():
    def __init__(self, device):
        self.parameter = torch.Tensor([0.3216, 0.8353, 0.5490])
        self.device = device

    def filter(self, x, alpha):
        (R, G, B) = alpha
        assert x.shape[-1] == 3
        new_x = torch.from_numpy(np.zeros_like(x)).to(torch.float32).to(self.device)
        xx = x.copy()
        r = xx[:, :, 0]
        g = xx[:, :, 1]
        b = xx[:, :, 2]
        gray = 0.3 * r + 0.59 * g + 0.11 * b
        ratio = torch.from_numpy(gray).to(self.device)
        new_x[:, :, 0] = ratio * R
        new_x[:, :, 1] = ratio * G
        new_x[:, :, 2] = ratio * B
        new_x = torch.clamp(new_x, 0, 255)
        return new_x
