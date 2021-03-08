import os
from jax import numpy as jnp

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

def im2float(im, dtype=jnp.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1
    :param im: image
    :param dtype: default jnp.float32
    :return im: image converted to specified dtype
    """
    if issubclass(im.dtype.type, jnp.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, jnp.integer):
        return im / dtype(jnp.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def pad_image(field, target_shape, padval=0):
    """
    :param field: input field
    :param target_shape: desired shape for output
    :param padval: value to pad the input field with
    :return field: output field with desired shape
    """
    size_diff = jnp.array(target_shape) - jnp.array(field.shape[-3:-1])
    odd_dim = jnp.array(field.shape[-3:-1]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = jnp.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        leading_dims = field.ndim - 3  # only pad the middle two dims
        if leading_dims > 0:
            pad_front = jnp.concatenate(([0] * leading_dims, pad_front, 0))
            pad_end = jnp.concatenate(([0] * leading_dims, pad_end, 0))
        return jnp.pad(field, tuple(zip(pad_front, pad_end)), 'constant',
                        constant_values=padval)
    else:
        return field


def crop_image(field, target_shape):
    """
    :param field: input field
    :param target_shape: desired shape for output
    :return field: output field with desired shape
    """
    size_diff = jnp.array(field.shape[-3:-1]) - jnp.array(target_shape)
    odd_dim = jnp.array(field.shape[-3:-1]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = jnp.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        return jnp.expand_dims(field[...,0][(..., *crop_slices)], axis=-1)
    else:
        return field