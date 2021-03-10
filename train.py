"""
Neural holography:

This is the main executive script used for training our parameterized wave propagation model

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

-----

$ python train_model_offline.py --channel=1
"""

import os
import cv2
import sys

sys.path.append('../cvhn')
import statistics
import torch
import utils
from phase_capture_loader import PhaseCaptureLoader
from tensorboardX import SummaryWriter
import train_helper as helper
from models import PropagationCNN
from flax import serialization, optim
from jax import jit
from skimage import io
import jax
from jax import random
from jax import numpy as jnp
from imageio import imread
import numpy as np

# Command line argument processing
p = helper.argument_parser()
opt = p.parse_args()
opt = helper.force_options(
    opt)  # set query with opt.experiment in command line

# Initialize Run ID
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
run_id = f'{chan_str}_{opt.experiment}_' \
         f'Target{opt.target_network}-Activation{opt.activation}-Norm{opt.norm}_' \
         f'{opt.loss_type}loss_lr{opt.lr_model}'
print(f'   - Training forward propagation model...')

# Initialize setup parameters
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = helper.prop_dist(opt.channel, opt.sled)  # propagation distances
wavelength = (636.4 * nm, 517.7 * nm, 440.8 * nm)[channel]

if opt.sled:
    wavelength = (634.8 * nm, 510 * nm, 450 * nm)[channel]
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)  # 1080p dataset
roi_res = (880, 1600)  # regions of interest (to penalize)

# Setup Loss Functions
roi_mask = jnp.ones((1, *image_res, 1), jnp.float32)
roi_mask = utils.pad_image(utils.crop_image(roi_mask, roi_res), image_res)

# Initialize model
im = imread(os.path.join(os.path.join(opt.phase_path, 'test'), "10_0.png"))
im = (1 - im / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
phase = jnp.array(torch.tensor(im, dtype=torch.float32).reshape(*im.shape,
                                                                1))[..., 0]
mode = helper.get_mode(opt.target_network)
print(f'Mode set: {mode}')
key = random.PRNGKey(0)
model = PropagationCNN(mode=mode, d=prop_dist)
variables = model.init(key, phase)


@jit
def apply(variables, phase):
    return model.apply(variables, phase)


# Setup optimizer
def create_optimizer(params, learning_rate=opt.lr_model):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


@jit
def mse(x, y):
    return jnp.mean(roi_mask * ((y - x)**2))


if opt.loss_type.lower() == 'l1':

    @jit
    def loss_train(x, y):
        return jnp.mean(roi_mask * jnp.abs(y - x))
elif opt.loss_type.lower() == 'l2':

    @jit
    def loss_train(x, y):
        return jnp.mean(roi_mask * ((y - x)**2))


# Path for data
model_path = opt.model_path  # path for new model checkpoints
utils.cond_mkdir(model_path)

# Load pretrained model and start from there
if opt.pretrained_path != '':
    print(f'   - Start from pre-trained model: {opt.pretrained_path}')
    ifile = open(opt.pretrained_path, 'rb')
    bytes_input = ifile.read()
    ifile.close()
    variables = serialization.from_bytes(variables, bytes_input)


# Setup Update step
@jit
def train_step(optimizer, batch):
    # Batch contains a single field, size (1, H, W, C)
    phase = batch['phase'][0, ..., 0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    def _loss(params):
        simulated = apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0),
                                    axis=-1)
        return loss_train(simulated, captured)

    # If returning more than loss from _loss, set has_aux=True
    # holomorphic=True if doing complex optimization (?)
    grad_fn = jax.value_and_grad(_loss)

    # out is the simulated result from our model, not necessary
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    return optimizer, loss


@jit
def val_step(optimizer, batch):
    # Batch contains a single field, size (1, H, W, C)
    phase = batch['phase'][0, ..., 0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    def _val_forward(params):
        simulated = apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0),
                                    axis=-1)
        return simulated, loss_train(simulated,
                                     captured), mse(simulated, captured)

    simulated, loss, loss_mse = _val_forward(optimizer.target)

    return simulated, loss, loss_mse


@jit
def compute_mse(optimizer, batch):
    # Batch contains a single field, size (1, H, W, C)

    # You can swap out the error function depending on the network type.
    phase = batch['phase'][0, ..., 0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    def _loss_mse(params):
        simulated = apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0),
                                    axis=-1)
        return mse(simulated, captured)

    loss_mse = _loss_mse(optimizer.target)
    return loss_mse


@jit
def get_predicted_amp(optimizer, batch):
    # Batch contains a single field, size (1, H, W, C)

    # You can swap out the error function depending on the network type.
    phase = batch['phase'][0, ..., 0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    model_amp = apply(optimizer.target, phase)
    model_amp = jnp.expand_dims(jnp.expand_dims(model_amp, axis=0), axis=-1)
    return model_amp


# Create Optimizer
optimizer = create_optimizer(variables, learning_rate=opt.lr_model)

# phase, captured images Loader
training_phases = ['train']
train_loader = torch.utils.data.DataLoader(PhaseCaptureLoader(
    os.path.join(opt.phase_path, 'train'),
    os.path.join(opt.captured_path, 'train'),
    channel=channel,
    image_res=image_res,
    shuffle=True,
    sled=opt.sled),
                                           batch_size=1)
loaders = {'train': train_loader}
# run validation every epoch.
training_phases.append('val')
loaders['val'] = torch.utils.data.DataLoader(PhaseCaptureLoader(
    os.path.join(opt.phase_path, 'val'),
    os.path.join(opt.captured_path, 'val'),
    channel=channel,
    image_res=image_res,
    shuffle=True,
    sled=opt.sled),
                                             batch_size=1)

# tensorboard writer
summaries_dir = os.path.join(opt.tb_path, run_id)
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(f'{summaries_dir}')
tensorboard_im_freq = 10000
i_acc = 0
for e in range(opt.num_epochs):
    print(f'   - Epoch {e+1} ...')
    running_losses = {}
    running_losses_mse = {}

    for phase in training_phases:
        loader = loaders[phase]
        running_loss = 0.
        running_loss_mse = 0.

        if phase == 'train':
            tensorboard_freq = 100
        else:
            psnr_list = []
            tensorboard_freq = 1

        for i, phase_capture in enumerate(loader):
            if i % tensorboard_freq == 0:
                print(f'   - Epoch {e+1}:{i}')

            # SLM phase, Captured amp(s), and idxs of corresponding planes
            slm_phase, captured_amp, captured_filename = phase_capture
            slm_phase = jnp.array(slm_phase)
            captured_amp = jnp.array(captured_amp)

            batch = {
                'phase': slm_phase,  # (H, W)
                'captured': captured_amp,  # (H, W)
            }

            if phase == 'train':
                optimizer, loss = train_step(optimizer, batch)
                if i % tensorboard_freq == 0:
                    loss_mse = compute_mse(optimizer, batch)
                if i % tensorboard_im_freq == 0 and opt.tb_image:
                    model_amp = get_predicted_amp(optimizer, batch)
            elif phase == 'val':
                model_amp, loss, loss_mse = val_step(optimizer, batch)

            # write to tensorboard
            if i % tensorboard_freq == 0:
                writer.add_scalar(f'Loss_{phase}/objective', np.array(loss),
                                  i_acc)
                writer.add_scalar(f'Loss_{phase}/L2', np.array(loss_mse),
                                  i_acc)
                print(captured_amp.shape)
                captured_amp = utils.crop_image(captured_amp, roi_res)

                if i % tensorboard_im_freq == 0 and opt.tb_image:
                    print(model_amp.shape, roi_res, captured_filename)
                    model_amp = utils.crop_image(model_amp, roi_res)
                    model_amp = model_amp[..., 0]
                    captured_amp = captured_amp[..., 0]
                    max_amp = max(model_amp.max(), captured_amp.max())
                    writer.add_image(f'{phase}/recon_{i}',
                                     np.array(model_amp / max_amp), i_acc)
                    writer.add_image(f'{phase}/captured_{i}',
                                     np.array(captured_amp / max_amp), i_acc)

                # Compute SRGB PSNR on GPU
                # psnr_srgb = helper.psnr_srgb(model_amp, captured_amp)

                # if phase == 'val':
                #     psnr_list.append(psnr_srgb.item())
                # writer.add_scalar(f'PSNR_srgb/{phase}', psnr_srgb,
                #                   i_acc)

            i_acc += 1
            running_loss += np.array(loss)
            running_loss_mse += np.array(loss_mse)

            running_losses[phase] = running_loss / len(
                loader)  # average loss over epoch
            running_losses_mse[phase] = running_loss_mse / len(
                loader)  # average mse loss over epoch
            break

    # report every epoch
    # from IPython import embed
    # embed()
    writer.add_scalars('Loss_per_epoch/objective', running_losses, e + 1)
    writer.add_scalars('Loss_per_epoch/L2', running_losses_mse, e + 1)

    # if phase == 'val':
    #     writer.add_scalar(f'Validation_PSNR_per_epoch/average',
    #                       statistics.mean(psnr_list), e + 1)
    #     writer.add_scalar(f'Validation_PSNR_per_epoch/std_dev',
    #                       statistics.stdev(psnr_list), e + 1)
    #     writer.add_scalar(f'Validation_PSNR_per_epoch/min', min(psnr_list),
    #                       e + 1)
    #     writer.add_scalar(f'Validation_PSNR_per_epoch/max', max(psnr_list),
    #                       e + 1)

    # save model, every epoch
    bytes_output = serialization.to_bytes(optimizer.target)
    ofile = open(os.path.join(model_path, f'{run_id}_model_{e+1}epoch.pth'),
                 'wb')
    ofile.write(bytes_output)
    ofile.close()
