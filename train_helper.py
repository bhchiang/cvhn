import numpy as np
import torch.nn as nn
import utils
import torch
import configargparse
from models import Mode


def argument_parser():
    """
    return p: argparser for options needed to train model
    """
    p = configargparse.ArgumentParser()
    p.add_argument('--channel',
                   type=int,
                   default=1,
                   help='red:0, green:1, blue:2, rgb:3')
    p.add_argument('--pretrained_path',
                   type=str,
                   default='',
                   help='Path of pretrained checkpoints as a starting point.')
    p.add_argument('--model_path',
                   type=str,
                   default='./models',
                   help='Directory for saving out checkpoints')
    p.add_argument('--tb_path',
                   type=str,
                   default='./runs',
                   help='Directory for tensorboard files')
    p.add_argument('--tb_image',
                   type=utils.str2bool,
                   default=True,
                   help='If false, do not add image on tb')
    p.add_argument(
        '--phase_path',
        type=str,
        default=None,
        help='Directory for pre-calculated phases, attach /train or /val')
    p.add_argument(
        '--captured_path',
        type=str,
        default=None,
        help='Directory of pre-captured images, attach /train or /val')
    p.add_argument('--lr_model',
                   type=float,
                   default=5e-4,
                   help='Learning rate for model parameters')
    p.add_argument('--num_epochs',
                   type=int,
                   default=350,
                   help='Number of epochs')
    p.add_argument('--experiment',
                   type=str,
                   default='',
                   help='Name of the experiment')
    p.add_argument('--target_network',
                   type=str,
                   default='CNNr',
                   help='Name of the Target Model')
    p.add_argument('--loss_type', type=str, default='L1', help='Loss type')
    p.add_argument('--outer_skip',
                   type=utils.str2bool,
                   default=True,
                   help='If true, add outer skip to UNet')
    p.add_argument('--norm',
                   type=str,
                   default='instance',
                   help='type of normalization layers')
    p.add_argument('--activation',
                   type=str,
                   default='relu',
                   help='type of activation layers')
    p.add_argument('--sled',
                   type=utils.str2bool,
                   default=False,
                   help='set wavelength for SLED')
    return p


def prop_dist(channel, sled=False):
    """
    :param channel: number indicating color (1: R, 2: G, 3: B)
    :param sled: Flag with whether to use SLED or not
    :return prop_dist: distance from SLM to the target plane
    """
    cm = 1e-2
    if sled:
        if channel == 0:
            prop_dist = 13.43 * cm
        elif channel == 1:
            prop_dist = 13.56 * cm
        elif channel == 2:
            prop_dist = 13.53 * cm
    else:
        if channel == 0:
            prop_dist = 13.2 * cm
        elif channel == 1:
            prop_dist = 13.33 * cm
        elif channel == 2:
            prop_dist = 13.37 * cm

    return prop_dist


def get_mode(target_network):
    """
    :param target_network: string with target network type
    :return mode
    """
    if 'cnnr' in target_network.lower():
        mode = Mode.AMPLITUDE
    elif 'stackedcnnc' in target_network.lower():
        mode = Mode.STACKED_COMPLEX
    elif 'complexcnnc' in target_network.lower():
        mode = Mode.COMPLEX
    return mode


def psnr_srgb(recon, target):
    """
    calculate psnr in srgb between reconstructed image and target image in the range [0, 1]
    :param recon: reconstructed image amplitude
    :param target: target image amplitude
    :return psnr: psnr in srgb space
    """
    max_amp = max(recon.max(), target.max())
    target_linear = (target / max_amp)**2
    recon_linear = (recon / max_amp)**2
    target_srgb = torch.clip(target_linear, 0.0, 1.0)
    thresh = 0.0031308
    target_srgb = (12.92 * target_srgb) * (target_srgb <= thresh) \
                  + (1.055 * (target_srgb ** (1 / 2.4)) - 0.055) * (~(target_srgb <= thresh))
    recon_srgb = torch.clip(recon_linear, 0.0, 1.0)
    recon_srgb = (12.92 * recon_srgb) * (recon_srgb <= thresh) \
                 + (1.055 * (recon_srgb ** (1 / 2.4)) - 0.055) * (~(recon_srgb <= thresh))
    return 10 * jnp.log10(1 / ((((target_srgb - recon_srgb)**2).mean())))


def force_options(opt):
    """
    :param opt: opt with parameters for training
    :return opt: opt with parameters forced to appropriate values for different networks
    """
    if 'cnnr' in opt.target_network.lower():
        opt.outer_skip = True
        opt.norm = 'instance'
        opt.activation = 'relu'
    elif 'stackedcnnc' in opt.target_network.lower():
        opt.outer_skip = True
        opt.norm = 'instance'
        opt.activation = 'relu'
    elif 'complexcnnc' in opt.target_network.lower():
        opt.outer_skip = True
    return opt
