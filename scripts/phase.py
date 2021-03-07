import torch
from skimage import io
from IPython import embed


def polar_to_rect(mag, ang):
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


phase = io.imread("../sample_pairs/phase/10_0.png")
phase = torch.Tensor(phase)
real, imag = polar_to_rect(torch.ones_like(phase), phase)
embed()