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
import asm
import statistics
import torch
import utils
from phase_capture_loader import PhaseCaptureLoader
from tensorboardX import SummaryWriter
import train_helper as helper
from models import PropagationCNN
from flax import serialization, optim
import optimize
from jax import jit
from skimage import io
import jax
from tqdm import tqdm
from jax import random
from jax import numpy as jnp
from imageio import imread
import numpy as np

# Command line argument processing
p = helper.argument_parser()
opt = p.parse_args()

# Initialize Run ID
channel = opt.channel  # Red:0 / Green:1 / Blue:2
_optimizer = opt.optimizer
chan_str = ('red', 'green', 'blue')[channel]
run_id = f'{chan_str}_{opt.experiment}_' \
         f'Target{opt.target_network}-Activation{opt.activation}-Norm{opt.norm}_' \
         f'Test_Evaluation'
print(f'   - Testing forward propagation model...')
print(run_id)

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
print(f"Outer skip: {opt.outer_skip, type(opt.outer_skip)}")
print(f"Activation: {opt.activation}")
activation = opt.activation
key = random.PRNGKey(0)
model = PropagationCNN(mode=mode,
                       d=prop_dist,
                       outer_skip=opt.outer_skip,
                       activation=opt.activation)
variables = model.init(key, phase)
print("Model created")


@jit
def apply(variables, phase):
    return model.apply(variables, phase)


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
else:
    raise ValueError("Pass in a pretrained model path for test evaluation.")


@jit
def val_step(variables, batch):
    # Batch contains a single field, size (1, H, W, C)
    phase = batch['phase'][0, ..., 0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    def _val_forward(params):
        simulated = apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0),
                                    axis=-1)
        return simulated, loss_train(simulated,
                                     captured), mse(simulated, captured)

    simulated, loss, loss_mse = _val_forward(variables)

    return simulated, loss, loss_mse


# phase, captured images Loader
loader = torch.utils.data.DataLoader(PhaseCaptureLoader(
    os.path.join(opt.phase_path, 'test'),
    os.path.join(opt.captured_path, 'test'),
    channel=channel,
    image_res=image_res,
    shuffle=False,
    sled=opt.sled),
                                     batch_size=1)

# tensorboard writer
summaries_dir = os.path.join(opt.tb_path, run_id)
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(f'{summaries_dir}')
tensorboard_im_count = 10

running_loss = 0.
running_loss_mse = 0.
H = None
for i, phase_capture in tqdm(enumerate(loader)):
    print(f'{i}')

    # SLM phase, Captured amp(s), and idxs of corresponding planes
    slm_phase, captured_amp, captured_filename = phase_capture
    slm_phase = jnp.array(slm_phase)
    captured_amp = jnp.array(captured_amp)

    batch = {
        'phase': slm_phase,
        'captured': captured_amp,
    }
    model_amp, loss, loss_mse = val_step(variables, batch)

    # Write to tensorboard
    writer.add_scalar(f'objective', np.array(loss), i)
    writer.add_scalar(f'L2', np.array(loss_mse), i)
    captured_amp = utils.crop_image(captured_amp, roi_res)

    if i % 300 < tensorboard_im_count and opt.tb_image:
        if H is None:
            H = asm.compute(image_res, feature_size, wavelength, prop_dist)

        slm_field = jnp.exp(slm_phase[0, ..., 0] * 1j)
        a_h, a_w = H.shape
        b_h, b_w = image_res
        pad_y = (a_h - b_h) // 2
        pad_x = (a_w - b_w) // 2
        slm_field = asm._pad(slm_field, pad_y, pad_x)
        z = asm.propagate(slm_field, H)
        z = asm._crop(z, pad_y, pad_x)
        ideal_amp = jnp.abs(
            jnp.expand_dims(jnp.expand_dims(z, axis=-1), axis=0))
        ideal_amp = utils.crop_image(ideal_amp, roi_res)
        ideal_amp = ideal_amp * captured_amp.mean() / ideal_amp.mean()

        model_amp = utils.crop_image(model_amp, roi_res)
        model_amp = model_amp[..., 0]
        captured_amp = captured_amp[..., 0]
        ideal_amp = ideal_amp[..., 0]
        max_amp = max(max(model_amp.max(), captured_amp.max()),
                      ideal_amp.max())
        writer.add_image(f'recon', np.array(model_amp / max_amp), i)
        writer.add_image(f'captured', np.array(captured_amp / max_amp), i)
        writer.add_image(f'ideal', np.array(ideal_amp / max_amp), i)

    running_loss += np.array(loss)
    running_loss_mse += np.array(loss_mse)

avg_objective = running_loss / len(loader)
avg_mse = running_loss_mse / len(loader)
print(running_loss, running_loss_mse)
print(f"running_loss = {running_loss}, running_loss_mse = {running_loss_mse}")
print(f"avg_mse = {avg_mse}, avg_objective = {avg_objective}")
writer.add_scalar('Avg_objective', avg_objective, 0)
writer.add_scalar('Avg_L2', avg_mse, 0)
