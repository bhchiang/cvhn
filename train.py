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
from models import PropCNN

# Command line argument processing
p = helper.argument_parser()
opt = p.parse_args()
opt = helper.force_options(opt)  # set query with opt.experiment in command line

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
run_id = f'{chan_str}_{opt.experiment}_' \
         f'Target{opt.target_network}-Activation{opt.activation}-Norm{opt.norm}_' \
         f'{opt.loss_type}loss_lr{opt.lr_model}' 

print(f'   - Training forward propagation model....')
# units
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

prop_dists = helper.prop_dist(opt.channel, opt.sled)  # propagation distances for all 11 planes
wavelength = (636.4 * nm, 517.7 * nm, 440.8 * nm)[channel]
if opt.sled:
    wavelength = (634.8 * nm, 510 * nm, 450 * nm)[channel]

feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)  # 1080p dataset
roi_res = (880, 1600)  # regions of interest (to penalize)

#### TODO: SETUP LOSS ##########
loss_model = helper.loss_model(opt.loss_type).to(device)
loss_mse = nn.MSELoss().to(device)

# Path for data
model_path = opt.model_path  # path for new model checkpoints
utils.cond_mkdir(model_path)

model = PropCNN(dist, feature_size=feature_size, wavelength=wavelength,
                image_res=image_res, outer_skip=opt.outer_skip, target_network=opt.target_network,
                norm=opt.norm, activation=opt.activation)
params = model.init_params()

#### TODO: SETUP LOAD MODEL ##########
# Load pretrained model and start from there
if opt.pretrained_path != '':
    print(f'   - Start from pre-trained model: {opt.pretrained_path}')
    checkpoint = torch.load(opt.pretrained_path)
    params = 

#### TODO: SETUP ADAM OPTIMIZER ##########
optimizer_model = optim.Adam(model.parameters(), lr=(opt.lr_model * opt.batch_size / param_step_period))

#### TODO: UPDATE LOADER ##########
# phase, captured images Loader
training_phases = ['train']
train_loader = PairsLoader(os.path.join(opt.phase_path, 'train'),
                                                       os.path.join(opt.captured_path, 'train'),
                                                       loader_focus_idxs, channel=channel, image_res=image_res,
                                                       shuffle=True, area_on_sensor=area_on_sensor,
                                                       num_planes=opt.load_num_planes),
                                                       batch_size=opt.batch_size)
loaders = {'train': train_loader}

# run validation every epoch.
training_phases.append('val')
loaders['val'] = PairsLoader(os.path.join(opt.phase_path, 'val'),
                                                        os.path.join(opt.captured_path, 'val'),
                                                        loader_focus_idxs, channel=channel, image_res=image_res,
                                                        shuffle=True, area_on_sensor=area_on_sensor),
                                                        batch_size=opt.batch_size)

# tensorboard writer
summaries_dir = os.path.join(opt.tb_path, run_id)
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(f'{summaries_dir}')

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
            captured_path = os.path.join(opt.captured_path, 'train')
            tensorboard_freq = 1000
        else:
            model.eval()  # Set model to evaluate mode
            captured_path = os.path.join(opt.captured_path, 'val')
            psnr_list = []
            tensorboard_freq = 1

        for i, phase_capture in enumerate(loader):
            if i % tensorboard_freq == 0:
                print(i)

            # SLM phase, Captured amp(s), and idxs of corresponding planes
            slm_phase, captured_amp = phase_capture
            captured_amp = utils.crop_image(captured_amp, roi_res)

            if phase == 'train':
                # propagate forward through the model and crop to roi
                model_amp = model.apply(slm_phase, params)  # Should return (1, 1, *image_res) shape amplitude
                model_amp = utils.crop_image(model_amp, target_shape=roi_res)

                ### TODO: UPDATE LOSS COMPUTATION AND BACKWARDS###
                # calculate loss and backpropagate to model parameters
                loss_value_model = loss_model(model_amp, captured_amp)
                loss_value_model.backward()
                with torch.no_grad():
                    loss_mse_value_model = loss_mse(model_amp, captured_amp)
                optimizer_model.step()
            elif phase == 'val':
                ### TODO: UPDATE LOSS COMPUTATION AND WITH NO GRADIENT GRAPH###
                with torch.no_grad():
                    # propagate forward through the model
                    model_amp = model.apply(slm_phase)  # Should return (1, 1, *image_res) shape amplitude
                    model_amp = utils.crop_image(model_amp, target_shape=roi_res)

                    # calculate loss and backpropagate to model parameters
                    loss_value_model = loss_model(model_amp, captured_amp)
                    loss_mse_value_model = loss_mse(model_amp, captured_amp)

            # write to tensorboard
            with torch.no_grad():
                if i % tensorboard_freq == 0:
                    writer.add_scalar(f'Loss_{phase}/objective', loss_value_model.item(), i_acc)
                    writer.add_scalar(f'Loss_{phase}/L2', loss_mse_value_model.item(), i_acc)

                    model_amp = model_amp[0, ...].unsqueeze(0)
                    captured_amp = captured_amp[0, ...].unsqueeze(0)
                    max_amp = max(model_amp.max(), captured_amp.max())

                    if i % 10000 == 0 and opt.tb_image:
                        writer.add_image(f'{phase}/recon_{f}', model_amp / max_amp, i_acc)
                        writer.add_image(f'{phase}/captured_{f}', captured_amp / max_amp, i_acc)

                    # Compute SRGB PSNR on GPU
                    psnr_srgb = helper.psnr_srgb(model_amp, captured_amp)

                    if phase == 'val':
                        psnr_list.append(psnr_srgb.item())
                    writer.add_scalar(f'PSNR_srgb/{phase}', psnr_srgb.item(), i_acc)

                i_acc += 1
                running_loss += loss_value_model.item()
                running_loss_mse += loss_mse_value_model.item()

            running_losses[phase] = running_loss / len(loader)  # average train loss over epoch
            running_losses_mse[phase] = running_loss_mse / len(loader)  # average mse loss over epoch

            # save model, every epoch
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
        if e % 1 == 0:
            torch.save(model.state_dict(), os.path.join(model_path, f'{run_id}_model_{e}epoch.pth'))

