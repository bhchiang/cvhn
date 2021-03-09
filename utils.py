import os
from jax import numpy as jnp
import numpy as np
from jax import jit

def cond_mkdir(path):
    """
    create directory if it does not already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1
    :param im: image
    :param dtype: default jnp.float32
    :return im: image converted to specified dtype
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def pad_image(field, target_shape, padval=0):
    """
    :param field: input field (N, H, W, C)
    :param target_shape: desired shape for output (H', W')
    :param padval: value to pad the input field with
    :return field: output field with desired shape (N, H', W', C)
    """
    size_diff = jnp.array(target_shape) - jnp.array(field.shape[-3:-1])
    odd_dim = jnp.array(field.shape[-3:-1]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = jnp.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        pad_front = jnp.array([0, *pad_front, 0])
        pad_end = jnp.array([0, *pad_end, 0])
        return jnp.pad(field, tuple(zip(pad_front, pad_end)), 'constant',
                        constant_values=padval)
    else:
        return field

def crop_image(field, target_shape):
    """
    :param field: input field (N, H, W, C)
    :param target_shape: desired shape for output (H', W')
    :return field: output field with desired shape (N, H', W', C)
    """
    size_diff = jnp.array(field.shape[-3:-1]) - jnp.array(target_shape)
    odd_dim = jnp.array(field.shape[-3:-1]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = jnp.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        return field[:, crop_front[0]:-crop_end[0], crop_front[1]:-crop_end[1], :]
    else:
        return field