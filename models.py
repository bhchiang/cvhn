"""Flax implementation of U-Net based on 'Learned Hardware-in-the-loop Phase
Retrieval for Holographic Near-Eye Displays': https://www.cs.unc.edu/~cpk/data/papers/HIL-holography-suppl.pdf.
"""

import enum
from typing import Any, Callable, Literal, Tuple

import asm
import jax
from flax import linen as nn
from flax import optim
from flax.nn import initializers
from IPython import embed
from jax import jit, lax
from jax import numpy as jnp
from jax import random
from skimage import io

# Add more options if necessary
Norm = Literal["instance"]
Activation = Literal["relu"]

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


class Mode(enum.Enum):
    # One input channel representing the amplitude of our phase (real).
    AMPLITUDE = 1

    # Two real input channels, one representing the real component and
    # the other representing the imaginary component of the complex phase.
    STACKED_COMPLEX = 2

    # True complex mode, 1 complex input channel.
    COMPLEX = 3


# TODO: Implement InstanceNorm instead of LayerNorm
class InstanceNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        pass


class ComplexLayerNorm(nn.Module):
    """Complex version of layer norm based on the source: https://flax.readthedocs.io/en/latest/_modules/flax/linen/normalization.html#LayerNorm

    Original documentation: Layer normalization (https://arxiv.org/abs/1607.06450).
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
    dtype: Any = jnp.complex64
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


class UNetSkipConnectionBlock(nn.Module):
    down_nc: int
    up_nc: int
    mode: Mode
    layer_index: int

    submodule: nn.Module = None
    norm_layer: Norm = "instance"
    activation: Activation = "relu"

    use_dropout: bool = False

    def setup(self):
        # TODO: fix padding consistency issues
        if mode == Mode.COMPLEX:
            self.down_conv = nn.Conv(
                features=self.down_nc,
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

            self.up_conv = nn.ConvTranspose(features=self.up_nc,
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
                features=self.down_nc,
                kernel_size=(5, 5),
                strides=(2, 2),
                # padding=((2, 2), (2, 2)),
            )
            self.down_norm = InstanceNorm()
            _leaky_relu = lambda x: nn.leaky_relu(x, negative_slope=0.2)
            self.down_activation = _leaky_relu

            self.up_conv = nn.ConvTranspose(
                features=self.up_nc,
                kernel_size=(4, 4),
                strides=(2, 2),
                # padding=((1, 1), (1, 1)),
            )
            self.up_norm = InstanceNorm()
            self.up_activation = _leaky_relu

    @nn.compact
    def __call__(self, x):
        print(
            f"\nSkip block, layer index = {self.layer_index}, down_nc = {self.down_nc}, up_nc = {self.up_nc}, x.shape = {x.shape}"
        )
        # embed()
        out = self.down_conv(x)
        print(f"After down conv, out.shape = {out.shape}")

        if not self.layer_index in [7, 8]:
            out = self.down_norm(out)

        out = self.down_activation(out)

        if self.submodule is not None:
            # print(f"Applying submodule recursively, out.shape = {out.shape}")
            out = self.submodule(out)

        out = self.up_conv(out)
        print(f"After up conv, out.shape = {out.shape}")

        if not self.layer_index in [7, 8]:
            out = self.up_norm(out)

        if not self.layer_index == 8:
            out = self.up_activation(out)

        if self.use_dropout:
            out = nn.Dropout(rate=0.5)

        print(f"Before skip: out.shape = {out.shape}, x.shape = {x.shape}")

        # Concatenate along the last axis (channels assuming NHWC)
        out = jnp.dstack((x, out))

        print(f"After skip: out.shape = {out.shape}, x.shape = {x.shape}\n")
        return out


class UNet(nn.Module):
    """
    We construct the U-Net from the innermost layer to the outermost layer recursively.
    """

    input_nc_target: int
    output_nc_target: int
    mode: Mode

    norm_layer: Norm = "instance"
    activation: Activation = "relu"
    outer_skip: bool = False

    # Order: innermost to outermost
    down: jnp.ndarray = jnp.array([512, 512, 512, 512, 512, 256, 128, 64])
    up: jnp.ndarray = jnp.array([512, 512, 512, 512, 256, 128, 64, 32])

    def setup(self):
        print(f"Mode = {self.mode}")
        """
        For the true complex network (Mode.COMPLEX), all layers must be explicity initialized with
        complex weights and biases.
        
        Ensure that there are no conversions to real (jax.lax will emit a ComplexWarning if there
        are any).

        Differences from the original implementation:
        1. We are not dealing with color here, so the number of input channels and output channels is 1.
        2. We are including a skip connection for the outermost layer.

        """

        unet = None
        for i, (_down, _up) in enumerate(zip(self.down, self.up)):
            unet = UNetSkipConnectionBlock(
                down_nc=_down,
                up_nc=_up,
                mode=self.mode,
                norm_layer=self.norm_layer,
                activation=self.activation,
                submodule=unet,
                layer_index=i + 1,
            )

        self.unet = unet

    @nn.compact
    def __call__(self, x):
        out = self.unet(x)

        # Add ending convolutional layer
        _complex = self.mode == Mode.COMPLEX

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


class PropagationCNN(nn.Module):
    z: float  # Propagation distance in meters
    mode: Mode  # Target network type

    feature_size: jnp.ndarray = jnp.array([6.4e-6] * 2)
    wavelength: float = 520e-9

    # Resolution of input SLM phase field
    slm_resolution: Tuple[int, int] = (1080, 1920)
    image_resolution: Tuple[int, int] = (1080, 1920)

    norm: Norm = 'instance'
    activation: Activation = 'relu'

    def setup(self):
        # Define number of input and output channels
        input_nc_target = output_nc_target = 1

        if mode == Mode.STACKED_COMPLEX:
            input_nc_target = output_nc_target = 2

        self.input_nc_target = input_nc_target
        self.output_nc_target = output_nc_target

        # Define ASM propagation kernel
        H = compute(self.input_resolution, self.feature_size, self.wavelength,
                    self.z)
        self.H = H

        # Define target U-Net
        unet = UNet(input_nc_target=input_nc_target,
                    output_nc_target=output_nc_target,
                    mode=mode)
        self.unet = unet
        pass

    @nn.compact
    def __call__(self, phase):
        slm_field = jnp.ones_like(phase) * jnp.exp(phase * 1j)
        slm_field = jnp.expand_dims(slm_field, axis=-1)

        # Do ASM propagation
        target_field = asm.propagate(slm_field, self.H)

        # Send through target U-Net
        if self.mode == Mode.AMPLITUDE:
            target_field = jnp.abs(target_field)
        elif self.mode == Mode.STACKED_COMPLEX:
            target_field = jnp.dstack((target_field.real, target_field.imag))
        elif self.mode == Mode.COMPLEX:
            pass

        # TODO: optional padding and cropping, for now we're doing padding = 'SAME'
        out = self.unet()
        pass


if __name__ == "__main__":

    # Example just to show the network runs
    phase = io.imread("sample_pairs/phase/10_0.png")
    captured = io.imread(
        "sample_pairs/captured/10_0_5.png")  # Intermediate plane

    # Running into OOM issues
    # Set to power of 2 to avoid running into shape issues while down or up sampling
    captured = captured[:512, :512]
    phase = phase[:512, :512]

    mode = Mode.COMPLEX
    print(phase.shape, phase.dtype)

    key = random.PRNGKey(0)

    model = PropagationCNN(mode=mode, z=0.05)
    variables = model.init(key, phase)

    embed()

    @jit
    def create_optimizer(params, learning_rate=0.001):
        optimizer_def = optim.Adam(learning_rate=learning_rate)
        optimizer = optimizer_def.create(params)
        return optimizer

    # Define additional error functions if desired
    @jit
    def mse(x, y):
        return jnp.abs(jnp.mean(y - x)**2)

    @jit
    def train_step(optimizer, batch, error=mse):
        # Batch contains a single field, size (1, H, W, C)

        # You can swap out the error function depending on the network type.
        phase = batch['phase']  # Input SLM phase
        captured = batch['captured']  # Label (amplitude)

        def _loss(params):
            simulated = model.apply(params, phase)
            return error(simulated, captured)

        # If returning more than loss from _loss, set has_aux=True
        # holomorphic=True if doing complex optimization (?)
        grad_fn = jax.value_and_grad(_loss)

        # out is the simulated result from our model, not necessary
        out, grad = grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, out

    batch = {
        'phase': phase,  # (H, W, 1)
        'captured': jnp.expand_dims(captured[:512, 512], axis=-1),
    }
    optimizer = create_optimizer(variables)
    optimizer, out = train_step(optimizer, batch)
    embed()
