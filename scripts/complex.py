from typing import Any, Callable, Tuple

from flax import linen as nn
from flax.nn import initializers
from IPython import embed
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?


class LayerNorm(nn.Module):
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
        print("======== 1")
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            mul = mul * jnp.asarray(
                self.param('scale', self.scale_init, (features, )), self.dtype)
        y = (x - mean) * mul
        if self.use_bias:
            y = y + jnp.asarray(
                self.param('bias', self.bias_init, (features, )), self.dtype)
        print("======== 2")
        return jnp.asarray(y, self.dtype)


def complex_kernel_init(rng, shape):
    print(shape)
    # Kernel is shape H x W x I x O
    # I = input channels
    # O = output channels
    fan_in = jnp.prod(jnp.array(shape[:-1]))
    x = random.normal(random.PRNGKey(0),
                      shape) + 1j * random.normal(random.PRNGKey(0), shape)
    return x * (2 * fan_in)**-0.5


def complex_bias_init(rng, shape):
    print("Initializing bias")
    print(rng, shape)
    return jnp.zeros(shape, jnp.complex64)


def complex_scale_init(rng, shape):
    print("scale init")
    print(rng, shape)
    return jnp.ones(shape, jnp.complex64)


class Model(nn.Module):
    features: int = 10

    def setup(self):
        self.conv = nn.Conv(features=self.features,
                            kernel_size=(2, 2),
                            strides=(1, 1),
                            dtype=jnp.complex64,
                            kernel_init=complex_kernel_init,
                            bias_init=complex_bias_init)
        self.norm = LayerNorm(dtype=jnp.complex64,
                              epsilon=jnp.complex64(1e-6),
                              bias_init=complex_bias_init,
                              scale_init=complex_scale_init)

    @nn.compact
    def __call__(self, x):
        embed()
        x = self.conv(x)
        print(x.dtype)
        x = self.norm(x)
        print(x.dtype, 'after norm')
        return x


k1, k2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(k1, (10, 5, 5, 3))
m = Model(features=10)
variables = m.init(k1, x)


# loss_fn = lambda vars: jnp.linalg.norm(m.apply(vars, x))**2
@jit
def loss(variables, x):
    y = m.apply(variables, x)


# grad(loss_fn)(variables)
embed()