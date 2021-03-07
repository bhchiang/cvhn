import enum
from typing import Any, Callable, Literal, Tuple

import asm
from flax import linen as nn
from flax.nn import initializers
from IPython import embed
from jax import jit, lax
from jax import numpy as jnp
from jax import random
from skimage import io

# Add more options if necessary (or convert to Enum)
Norm = Literal["instance"]
Activation = Literal["relu"]

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?


class Mode(enum.Enum):
    # One input channel representing the amplitude of our phase.
    AMPLITUDE = 1

    # Two real input channels, one representing the real component and
    # the other representing the imaginary component of the complex phase.
    STACKED_COMPLEX = 2

    # True complex, 1 complex input channel.
    COMPLEX = 3


# TODO: Implement InstanceNorm, using LayerNorm for now (both batch independent)
class InstanceNorm(nn.Module):
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(dtype=self.dtype)(x)


class ComplexLayerNorm(nn.Module):
    """
    Complex version of layer norm based on the source: https://flax.readthedocs.io/en/latest/_modules/flax/linen/normalization.html#LayerNorm
    """
    """Layer normalization (https://arxiv.org/abs/1607.06450).
        Operates on the last axis of the input data.

        It normalizes the activations of the layer for each given example in a
        batch independently, rather than across a batch like Batch Normalization.
        i.e. applies a transformation that maintains the mean activation within
        each example close to 0 and the activation standard deviation close to 1.

        Attributes:
            epsilon: A small float added to variance to avoid dividing by zero.
            dtype: the dtype of the computation (default: float32).
            use_bias:  If True, bias (beta) is added.
            use_scale: If True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
            bias_init: Initializer for bias, by default, zero.
            scale_init: Initializer for scale, by default, one.
    """
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input.

        Args:
        x: the inputs

        Returns:
        Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.complex64)
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        var = mean2 - lax.square(mean)
        mul = lax.rsqrt(var + jnp.complex64(self.epsilon))
        if self.use_scale:
            mul = mul * jnp.asarray(
                self.param('scale', self.scale_init, (features, )), self.dtype)
        y = (x - mean) * mul
        if self.use_bias:
            y = y + jnp.asarray(
                self.param('bias', self.bias_init, (features, )), self.dtype)
        return jnp.asarray(y, self.dtype)


# TODO: Add support for different types of initializers
def _complex_kernel_init(rng, shape, initializer=None):
    # Kernel is shape H x W x I x O
    # I = input channels
    # O = output channels

    # print(shape)
    # print("Initializing complex kernel")
    fan_in = jnp.prod(jnp.array(shape[:-1]))
    x = random.normal(random.PRNGKey(0),
                      shape) + 1j * random.normal(random.PRNGKey(0), shape)
    return x * (2 * fan_in)**-0.5


def _complex_bias_init(rng, shape):
    # print("Initializing complex bias")
    # print(rng, shape)
    return jnp.zeros(shape, jnp.complex64)


def _complex_scale_init(rng, shape):
    return jnp.ones(shape, jnp.complex64)


# TODO: implement other non-linearities
def _complex_relu(x):
    return (x.real > 0) * x


complex_extra = {
    'kernel_init': _complex_kernel_init,
    'bias_init': _complex_bias_init
}


class UNetSkipConnectionBlock(nn.Module):
    outer_nc: int
    inner_nc: int
    mode: Mode

    input_nc: int = -1

    innermost: bool = False
    outermost: bool = False

    submodule: nn.Module = None
    outer_skip: bool = False
    norm_layer: Norm = "instance"

    use_dropout: bool = False

    def setup(self):
        # input_nc = self.input_nc if self.input_nc < -1 else self.outer_nc
        _complex = mode == Mode.COMPLEX

        # TODO: fix padding consistency issues
        if _complex:
            self.down_conv = nn.Conv(
                features=self.inner_nc,
                kernel_size=(5, 5),
                strides=(2, 2),
                dtype=jnp.complex64,
                # padding=((2, 2), (2, 2)),
                kernel_init=_complex_kernel_init,
                bias_init=_complex_bias_init)
            self.down_norm = ComplexLayerNorm(dtype=jnp.complex64,
                                              bias_init=_complex_bias_init,
                                              scale_init=_complex_scale_init)
            self.down_activation = _complex_relu

            self.up_conv = nn.ConvTranspose(features=self.outer_nc,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            dtype=jnp.complex64,
                                            kernel_init=_complex_kernel_init,
                                            bias_init=_complex_bias_init)
            self.up_norm = ComplexLayerNorm(dtype=jnp.complex64,
                                            bias_init=_complex_bias_init,
                                            scale_init=_complex_scale_init)
            self.up_activation = _complex_relu
        else:
            self.down_conv = nn.Conv(
                features=self.inner_nc,
                kernel_size=(5, 5),
                strides=(2, 2),
                # padding=((2, 2), (2, 2)),
            )
            self.down_norm = InstanceNorm()
            _leaky_relu = lambda x: nn.leaky_relu(x, negative_slope=0.2)
            self.down_activation = _leaky_relu

            self.up_conv = nn.ConvTranspose(
                features=self.outer_nc,
                kernel_size=(4, 4),
                strides=(2, 2),
                # padding=((1, 1), (1, 1)),
            )
            self.up_norm = InstanceNorm()
            self.up_activation = _leaky_relu

    @nn.compact
    def __call__(self, x):
        print(
            f"Skip connection, outer_nc = {self.outer_nc}, inner_nc = {self.inner_nc}, x.shape = {x.shape}"
        )
        # embed()
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
    outer_ncs: dict
    inner_ncs: dict
    mode: Mode

    norm_layer: Norm = "instance"
    activation: Activation = "relu"
    outer_skip: bool = False

    # Legacy
    num_downs: int = 4
    nf0: int = 32
    max_channels: int = 512

    def setup(self):
        print(f"Mode = {self.mode}")

        # For the true complex network (Mode.COMPLEX), all layers must be explicity initialized with
        # complex weights and biases.
        #
        # Ensure that there are no conversions to real (jax.lax will emit a ComplexWarning).

        # Add innermost block
        print("Adding innnermost block")
        unet = UNetSkipConnectionBlock(
            outer_nc=self.outer_ncs['innermost'],
            inner_nc=self.inner_ncs['innermost'],
            # outer_nc=self.channels(self.num_downs - 1),
            # inner_nc=self.channels(self.num_downs - 1),
            norm_layer=self.norm_layer,
            innermost=True,
            mode=self.mode)

        # Recursively generate the UNet
        print("Starting loop")
        for outer_nc, inner_nc in zip(outer_ncs['middle'],
                                      inner_ncs['middle']):
            # outer_nc = self.channels(i)
            # inner_nc = self.channels(i + 1)
            # print(f"i = {i}, outer_nc = {outer_nc}, inner_nc = {inner_nc}")
            unet = UNetSkipConnectionBlock(
                outer_nc=outer_nc,
                inner_nc=inner_nc,
                submodule=unet,
                mode=self.mode,
            )

        # Add outermost block
        print("Adding outermost block")
        unet = UNetSkipConnectionBlock(
            outer_nc=self.outer_ncs['outermost'],
            inner_nc=self.inner_ncs['outermost'],
            #    outer_nc=min(self.nf0,
            #                 self.max_channels),
            #    inner_nc=min(2 * self.nf0,
            #                 self.max_channels),
            input_nc=self.input_nc_target,
            outermost=True,
            outer_skip=self.outer_skip,
            norm_layer=None,
            submodule=unet,
            mode=self.mode)
        self.unet = unet

    @nn.compact
    def __call__(self, x):
        out = self.unet(x)

        # Add additional convolutional layer
        _complex = self.mode == Mode.COMPLEX
        return out
        if _complex:
            out = nn.Conv(
                features=self.output_nc_target,
                kernel_size=(4, 4),
                strides=(1, 1),
                dtype=jnp.complex64,
                kernel_init=_complex_kernel_init,
                bias_init=_complex_bias_init,
                # padding=((2, 2), (2, 2))
            )(out)
        else:
            out = nn.Conv(
                features=self.output_nc_target,
                kernel_size=(4, 4),
                strides=(1, 1),
                # padding=((2, 2), (2, 2))
            )(out)

        return out

    @classmethod
    def inds(cls):
        return jnp.arange(cls.num_downs - 1, 1, -1)

    @classmethod
    def generate_ncs(cls):
        inds = cls.inds()

        outer_nc = {'middle': jnp.zeros(len(inds), jnp.int32)}
        inner_nc = {'middle': jnp.zeros(len(inds), jnp.int32)}

        inner_nc['innermost'] = cls.channels(cls.num_downs - 1)
        outer_nc['innermost'] = inner_nc['innermost']

        for j, i in enumerate(inds):
            outer_nc['middle'] = outer_nc['middle'].at[j].set(cls.channels(i))
            inner_nc['middle'] = inner_nc['middle'].at[j].set(
                cls.channels(i + 1))

        inner_nc['outermost'] = jnp.minimum(cls.nf0, cls.max_channels)
        outer_nc['outermost'] = jnp.minimum(2 * cls.nf0, cls.max_channels)
        return outer_nc, inner_nc

    @classmethod
    def channels(cls, n):
        return jnp.int32(jnp.minimum(2**n * cls.nf0, cls.max_channels))


if __name__ == "__main__":

    # Example just to show the network runs, doesn't actually do propagation / processing
    _phase = io.imread("sample_pairs/phase/10_0.png")

    phase = jnp.exp(1j * _phase)
    phase = jnp.expand_dims(phase, axis=-1)

    # Running into OOM issues
    # Set to power of 2 to avoid running into shape issues while down or up sampling
    phase = phase[:256, :128]
    # Tested to work for both square and non-square inputs
    mode = Mode.COMPLEX
    if mode == Mode.AMPLITUDE:
        phase = jnp.abs(phase)
    elif mode == Mode.STACKED_COMPLEX:
        phase = jnp.dstack((phase.real, phase.imag))
    elif mode == Mode.COMPLEX:
        pass

    print(phase.shape, phase.dtype)

    key = random.PRNGKey(0)
    input_nc_target = output_nc_target = 2 if mode == Mode.STACKED_COMPLEX else 1
    outer_ncs, inner_ncs = UNetGenerator.generate_ncs()
    embed()
    model = UNetGenerator(input_nc_target=input_nc_target,
                          output_nc_target=output_nc_target,
                          outer_ncs=outer_ncs,
                          inner_ncs=inner_ncs,
                          mode=mode)

    params = model.init(key, phase)

    # print(params.keys())

    gt = io.imread("sample_pairs/captured/10_0_5.png")  # Intermediate plane
    gt = gt + 0j

    @jit
    def _error(params, phase):
        y = model.apply(params, phase)
        _gt = jnp.expand_dims(gt[:256, :128], axis=-1)
        return jnp.abs(jnp.mean(y - _gt)**2)

    embed()
