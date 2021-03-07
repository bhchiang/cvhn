import utils
from asm import propagate, compute
from unet_generator import UnetGenerator
from jax import numpy as jnp
from flax import linen as nn

class PropCNN(nn.Module):
    """Creates Propagation Module that uses Propagation ASM followed by an Unet at target wavefront

    :param prop_dist:
    :param feature_size:
    :param wavelength:
    :param image_res:
    :param outer_skip:
    :param target_network:
    :param norm:
    :param activation:
    """
    def __init__(self, prop_dist, feature_size=(6.4e-6, 6.4e-6), wavelength=520e-9,
                 image_res=(1080, 1920), outer_skip=False, target_network='CNNr',
                 norm='instance', activation='relu'):

        super(PropCNN, self).__init__()

        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomped_H = compute(slm_field, self.feature_size, self.wavelength, self.prop_dist, kernel_size=self.kernel_size)
        self.kernel_size = -1
        self.image_res = image_res

        # The resolution of input for each network - to be multiplier of 2**num_downs
        self.unet_res_target = tuple(res if res % (2**8) == 0 else
                                     res + (2**8 - res % (2**8))
                                     for res in self.image_res)

        self.target_network_type = target_network.lower()
        if 'cnnr' in target_network.lower() or 'complexcnnc' in target_network.lower():
            input_nc_target = 1
            output_nc_target = 1
        elif 'stackedcnnc' in target_network.lower():
            input_nc_target = 2
            output_nc_target = 2
  
        if 'none' in target_network.lower():
            self.unet_target = None
            self.target_scale = 1.0
        else:
            self.target_scale = None
            self.unet_target = UnetGenerator(input_nc_target, output_nc_target, norm_layer=norm, activation=activation, outer_skip=outer_skip)

    
    def __call__(self, phase, params):
        # convert phase to real and imaginary
        slm_field = jnp.ones_like(phase) * jnp.exp(phase * 1j)

        # ASM prop from SLM plane to intermediate plane
        target_field = propagate(slm_field, self.precomped_H)

        # Correct output at target plane
        if self.unet_target is None:
            #### TODO: SET target_scale as trainable ####
            amp = params*jnp.abs(target_field)
        else:
            if 'complexcnnc' in self.target_network_type:
                input_unet_target = target_field  # Complex field to target network
            elif self.target_network_type == 'cnnr':
                input_unet_target = jnp.abs(target_field)  # Amplitude to target network
            elif self.target_network_type == 'stackedcnnc':
                input_unet_target = jnp.stack((jnp.real(target_field), jnp.imag(target_field)), axis=1)  # (real, imag) stacked channel

            # Pad for target unet, send through unet, and then crop output
            input_unet_target = utils.pad_image(input_unet_target, self.unet_res_target)
            output_unet_target = self.unet_target.forward(input_unet_target, params)
            output_unet_target = utils.crop_image(output_unet_target, self.image_res)
            
            if 'complexcnnc' in self.target_network_type:
                amp = jnp.abs(output_unet_target) # (real, imag) stacked channel
            elif self.target_network_type == 'cnnr':
                amp = output_unet_target  # Amplitude out of target network
            elif self.target_network_type == 'stackedcnnc':
                amp = jnp.sqrt(output_unet_target[:,0:1,...]**2+output_unet_target[:,1:2,...]**2)  # Amp from (real, imag) stacked channel 

        return amp


    def init_params():
        if self.unet_target is None:
            return 1.0
        else:
            return self.unet_target.init_params()
