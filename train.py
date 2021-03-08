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

import utils
from phase_capture_loader import PhaseCaptureLoader
from tensorboardX import SummaryWriter
import train_helper as helper
from models import PropagationCNN
from flax import serialization

# Command line argument processing
p = helper.argument_parser()
opt = p.parse_args()
opt = helper.force_options(opt)  # set query with opt.experiment in command line

# Initialize Run ID
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
run_id = f'{chan_str}_{opt.experiment}_' \
         f'Target{opt.target_network}-Activation{opt.activation}-Norm{opt.norm}_' \
         f'{opt.loss_type}loss_lr{opt.lr_model}' 
print(f'   - Training forward propagation model...')

# Initialize setup parameters
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dists = helper.prop_dist(opt.channel, opt.sled)  # propagation distances
wavelength = (636.4 * nm, 517.7 * nm, 440.8 * nm)[channel]
if opt.sled:
    wavelength = (634.8 * nm, 510 * nm, 450 * nm)[channel]
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)  # 1080p dataset
roi_res = (880, 1600)  # regions of interest (to penalize)

# Setup Loss Functions
@jit
def mse(x, y):
    return jnp.mean((y - x)**2)
@jit
def L1(x, y):
    return jnp.mean(jnp.abs(y - x))

# Path for data
model_path = opt.model_path  # path for new model checkpoints
utils.cond_mkdir(model_path)

# Initialize model
phase = io.imread("sample_pairs/phase/10_0.png")
captured = io.imread(
    "sample_pairs/captured/10_0_5.png")  # Intermediate plane
mode = Mode.COMPLEX
key = random.PRNGKey(0)
model = PropagationCNN(mode=mode, d=0.05)
variables = model.init(key, phase)
@jax.jit
def apply(variables, phase):
    return model.apply(variables, phase)

# Load pretrained model and start from there
if opt.pretrained_path != '':
    print(f'   - Start from pre-trained model: {opt.pretrained_path}')
    ifile = open(opt.pretrained_path, 'rb')
    bytes_input = ifile.read()
    ifile.close()
    variables = serialization.from_bytes(variables, bytes_input)

# Setup Optimizer
@jit
def create_optimizer(params, learning_rate=0.001):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer
optimizer = create_optimizer(variables)

# Setup Update step
@jit
def train_step(optimizer, batch, error=mse, update=True, return_amp=False, compute_mse=False):
    # Batch contains a single field, size (1, H, W, C)

    # You can swap out the error function depending on the network type.
    phase = batch['phase'][0,...,0]  # Input SLM phase
    captured = batch['captured']  # Label (amplitude)

    def _loss(params):
        simulated = model.apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0), axis=-1)
        simulated = utils.crop_image(simulated, roi_res)
        return error(simulated, captured)

    def _loss_mse(params):
        simulated = model.apply(params, phase)
        simulated = jnp.expand_dims(jnp.expand_dims(simulated, axis=0), axis=-1)
        simulated = utils.crop_image(simulated, roi_res)
        return mse(simulated, captured)

    if update:
        # If returning more than loss from _loss, set has_aux=True
        # holomorphic=True if doing complex optimization (?)
        grad_fn = jax.value_and_grad(_loss)

        # out is the simulated result from our model, not necessary
        loss, grad = grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
    else:
        loss = _loss(params)
    if compute_mse:
        loss_mse = _loss_mse(params)
    else:
        loss_mse = None
    if return_amp:
        model_amp = model.apply(params, phase)
        model_amp = jnp.expand_dims(jnp.expand_dims(model_amp, axis=0), axis=-1)
        model_amp = utils.crop_image(model_amp, roi_res)
    else:
        model_amp = None

    return optimizer, loss, loss_mse, model_amp

# phase, captured images Loader
training_phases = ['train']
train_loader = torch.utils.data.DataLoader(PairsLoader(os.path.join(opt.phase_path, 'train'),
                                                       os.path.join(opt.captured_path, 'train'),
                                                       channel=channel, image_res=image_res,
                                                       shuffle=True, sled=opt.sled), batch_size=1,
                                                       num_workers=8)
loaders = {'train': train_loader}
# run validation every epoch.
training_phases.append('val')
loaders['val'] = torch.utils.data.DataLoader(PairsLoader(os.path.join(opt.phase_path, 'val'),
                                                        os.path.join(opt.captured_path, 'val'),
                                                       channel=channel, image_res=image_res,
                                                       shuffle=True, sled=opt.sled), batch_size=1,
                                                       num_workers=8)

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
            model.train()  # Set model to training mode
            tensorboard_freq = 100
        else:
            model.eval()  # Set model to evaluate mode
            psnr_list = []
            tensorboard_freq = 1

        for i, phase_capture in enumerate(loader):
            if i % tensorboard_freq == 0:
                print(f'   - Epoch {e+1}:{i}')

            # SLM phase, Captured amp(s), and idxs of corresponding planes
            slm_phase, captured_amp = phase_capture
            slm_phase = jnp.array(slm_phase)
            captured_amp = jnp.array(captured_amp)
            captured_amp = utils.crop_image(captured_amp, roi_res)
            batch = {
                'phase': slm_phase,  # (H, W)
                'captured': captured_amp,  # (H, W)
            }

            return_amp = i % tensorboard_im_freq == 0 and opt.tb_image
            compute_mse = i % tensorboard_freq == 0
            if phase == 'train':
                optimizer, loss, loss_mse, model_amp = train_step(optimizer, batch, update=True,
                                                        compute_mse=compute_mse, return_amp=return_amp)
            elif phase == 'val':
                optimizer, loss, loss_mse, model_amp = train_step(optimizer, batch, update=False,
                                                        compute_mse=compute_mse, return_amp=return_amp)

            # write to tensorboard
            with torch.no_grad():
                if i % tensorboard_freq == 0:
                    writer.add_scalar(f'Loss_{phase}/objective', loss, i_acc)
                    writer.add_scalar(f'Loss_{phase}/L2', loss_mse, i_acc)

                    model_amp = model_amp[...,0]
                    captured_amp = captured_amp[...,0]
                    max_amp = max(model_amp.max(), captured_amp.max())

                    if i % tensorboard_im_freq == 0 and opt.tb_image:
                        writer.add_image(f'{phase}/recon_{f}', model_amp / max_amp, i_acc)
                        writer.add_image(f'{phase}/captured_{f}', captured_amp / max_amp, i_acc)

                    # Compute SRGB PSNR on GPU
                    psnr_srgb = helper.psnr_srgb(model_amp, captured_amp)

                    if phase == 'val':
                        psnr_list.append(psnr_srgb.item())
                    writer.add_scalar(f'PSNR_srgb/{phase}', psnr_srgb.item(), i_acc)

                i_acc += 1
                running_loss += loss
                running_loss_mse += loss_mse

            running_losses[phase] = running_loss / len(loader)  # average loss over epoch
            running_losses_mse[phase] = running_loss_mse / len(loader)  # average mse loss over epoch

        with torch.no_grad():
            # report every epoch
            writer.add_scalars('Loss_per_epoch/objective', running_losses, e + 1)
            writer.add_scalars('Loss_per_epoch/L2', running_losses_mse, e + 1)

            if phase == 'val':
                writer.add_scalar(f'Validation_PSNR_per_epoch/average', statistics.mean(psnr_list), e + 1)
                writer.add_scalar(f'Validation_PSNR_per_epoch/std_dev', statistics.stdev(psnr_list), e + 1)
                writer.add_scalar(f'Validation_PSNR_per_epoch/min', min(psnr_list), e + 1)
                writer.add_scalar(f'Validation_PSNR_per_epoch/max', max(psnr_list), e + 1)

        # save model, every epoch
        bytes_output = serialization.to_bytes(optimizer.target)
        ofile = open(os.path.join(model_path, f'{run_id}_model_{e+1}epoch.pth'), 'wb')
        ofile.write(byte_output)
        ofile.close()

