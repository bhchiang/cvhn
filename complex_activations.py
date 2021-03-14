from jax import numpy as jnp
from flax import linen as nn


def real_ReLU(x):
    return (x.real > 0) * x


def complex_ReLU(x):
    return (x.real > 0) * (x.imag > 0) * x


def complex_Cardiod(x):
    return 0.5 * (1 + x.real / (jnp.abs(x))) * x


def mod_ReLU(x, b=-1):
    return jnp.clip(jnp.abs(x) + b, a_min=0.0) / (jnp.abs(x)) * x


def _b_init(rng, shape):
    return -1.


class _mod_ReLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        b = self.param('b', _b_init, ())
        return mod_ReLU(x, b)


_compelex_activations = {
    "real_relu": real_ReLU,
    "complex_relu": complex_ReLU,
    "complex_cardiod": complex_Cardiod,
    "mod_relu": _mod_ReLU(),
}
