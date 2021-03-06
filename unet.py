from typing import Literal

import asm
from flax import linen as nn
from IPython import embed
from jax import numpy as jnp
from jax import random
from skimage import io
import enum

# Add more options if necessary (or convert to Enum)
Norm = Literal["instance"]
Activation = Literal["relu"]


class Mode(enum.Enum):
    # One input channel representing the amplitude of our phase.
    REAL_AMPLITUDE = 1

    # Two real input channels, one representing the real component and
    # the other representing the imaginary component of the complex phase.
    REAL_COMPLEX = 2

    # True complex, 1 complex input channel.
    COMPLEX = 3


# TODO: Implement InstanceNorm, using LayerNorm for now (both batch independent)
class InstanceNorm(nn.Module):
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(dtype=self.dtype)(x)


# TODO: Add support for different types of initializers
def _complex_kernel_init(rng, shape, initializer=None):
    # Kernel is shape H x W x I x O
    # I = input channels
    # O = output channels

    # print(shape)
    print("Initializing complex kernel")
    fan_in = jnp.prod(jnp.array(shape[:-1]))
    x = random.normal(random.PRNGKey(0),
                      shape) + 1j * random.normal(random.PRNGKey(0), shape)
    return x * (2 * fan_in)**-0.5


def _complex_bias_init(rng, shape):
    print("Initializing complex bias")
    # print(rng, shape)
    return jnp.zeros(shape, jnp.complex64)


def _complex_relu(x):
    return (x.real > 0) * x
    pass


complex_extra = {
    'kernel_init': _complex_kernel_init,
    'bias_init': _complex_bias_init
}


class UNetSkipConnectionBlock(nn.Module):
    outer_nc: int
    inner_nc: int
    mode: Mode
    dtype: jnp.dtype = jnp.float32

    input_nc: int = -1

    innermost: bool = False
    outermost: bool = False

    submodule: nn.Module = None
    outer_skip: bool = False
    norm_layer: Norm = "instance"

    use_dropout: bool = False

    def setup(self):
        # input_nc = self.input_nc if self.input_nc < -1 else self.outer_nc

        extra = complex_extra if mode == Mode.COMPLEX else {}
        _leaky_relu = lambda x: nn.leaky_relu(x, negative_slope=0.2)

        # TODO: fix padding consistency issues
        self.down_conv = nn.Conv(
            features=self.inner_nc,
            kernel_size=(5, 5),
            strides=(2, 2),
            dtype=self.dtype,
            # padding=((2, 2), (2, 2)),
            **extra)
        self.down_activation = _leaky_relu
        self.down_norm = InstanceNorm(dtype=self.dtype)

        self.up_conv = nn.ConvTranspose(
            features=self.outer_nc,
            kernel_size=(4, 4),
            strides=(2, 2),
            # padding=((1, 1), (1, 1)),
            **extra)
        self.up_norm = InstanceNorm()
        self.up_activation = _leaky_relu

    @nn.compact
    def __call__(self, x):
        print(
            f"Skip connection, outer_nc = {self.outer_nc}, inner_nc = {self.inner_nc}, x.shape = {x.shape}"
        )
        out = x
        out = self.down_conv(x)
        print(f"After down conv, out.shape = {out.shape}")

        if self.norm_layer is not None:
            out = self.down_norm(out)

        out = self.down_activation(out)

        if self.submodule is not None:
            print(f"Applying submodule recursively, out.shape = {out.shape}")
            out = self.submodule(out)

        out = self.up_conv(out)
        print(f"After up conv, out.shape = {out.shape}")

        if not self.outermost:
            if self.norm_layer is not None:
                out = self.up_norm(out)
            out = self.up_activation(out)

        if not self.innermost or not self.outermost:
            if self.use_dropout:
                out = nn.Dropout(rate=0.5)

        print(f"Before skip: out.shape = {out.shape}, x.shape = {x.shape}")

        if self.outermost and self.outer_skip is False:
            return out
        else:
            # Concatenate along the last axis (channels assuming NHWC)
            out = jnp.dstack((x, out))

        print(f"After skip: out.shape = {out.shape}, x.shape = {x.shape}")
        return out


class UNetGenerator(nn.Module):
    """
    We construct the U-Net from the innermost layer to the outermost layer recursively.
    """

    input_nc_target: int
    output_nc_target: int
    mode: Mode

    norm_layer: Norm = "instance"
    activation: Activation = "relu"
    outer_skip: bool = False

    # Legacy
    num_downs: int = 4
    nf0: int = 32
    max_channels: int = 512

    def setup(self):
        print(f"Mode = {mode}")
        # Add innermost block
        print("Adding innnermost block")
        unet = UNetSkipConnectionBlock(
            outer_nc=self.channels(self.num_downs - 1),
            inner_nc=self.channels(self.num_downs - 1),
            norm_layer=self.norm_layer,
            innermost=True,
            mode=mode,
        )

        # Recursively generate the UNet
        # print("Starting loop")
        # for i in jnp.arange(1, self.num_downs - 1)[::-1]:
        #     outer_nc = self.channels(i)
        #     inner_nc = self.channels(i + 1)
        #     print(f"i = {i}, outer_nc = {outer_nc}, inner_nc = {inner_nc}")
        #     unet = UNetSkipConnectionBlock(
        #         outer_nc=outer_nc,
        #         inner_nc=inner_nc,
        #         submodule=unet,
        #         mode=mode,
        #     )

        # # Add outermost block
        # print("Adding outermost block")
        # unet = UNetSkipConnectionBlock(outer_nc=min(self.nf0,
        #                                             self.max_channels),
        #                                inner_nc=min(2 * self.nf0,
        #                                             self.max_channels),
        #                                input_nc=self.input_nc_target,
        #                                outermost=True,
        #                                outer_skip=self.outer_skip,
        #                                norm_layer=None,
        #                                submodule=unet,
        #                                mode=mode)
        self.unet = unet

    @nn.compact
    def __call__(self, x):
        out = self.unet(x)

        # Add additional convolutional layer
        extra = complex_extra if mode == Mode.COMPLEX else {}
        out = nn.Conv(
            features=self.output_nc_target,
            kernel_size=(4, 4),
            strides=(1, 1),
            # padding=((2, 2), (2, 2))
            **extra,
        )(out)

        return out

    def channels(self, n):
        return min(2**n * self.nf0, self.max_channels)


if __name__ == "__main__":
    _phase = io.imread("sample_pairs/phase/10_0.png")
    phase = jnp.exp(1j * _phase)
    phase = jnp.expand_dims(phase, axis=-1)

    # Running into OOM issues
    # Set to power of 2 to avoid running into shape issues while down or up sampling
    phase = phase[:256, :128]
    # Tested to work for both square and non-square inputs
    mode = Mode.COMPLEX
    if mode == Mode.REAL_AMPLITUDE:
        phase = jnp.abs(phase)
    elif mode == Mode.REAL_COMPLEX:
        phase = jnp.dstack((phase.real, phase.imag))
    elif mode == Mode.COMPLEX:
        pass

    print(phase.shape, phase.dtype)

    key = random.PRNGKey(0)
    model = UNetGenerator(input_nc_target=1, output_nc_target=1, mode=mode)

    params = model.init(key, phase)
    # print(params.keys())
    # y = model.apply(params, phase)
    embed()
