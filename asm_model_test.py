from typing import Any, Callable, Tuple

from flax import linen as nn
from flax.nn import initializers
from IPython import embed
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import random
import asm


class Model(nn.Module):

    z: float = 0.05
    wavelength: float = 520e-9
    feature_size: float = jnp.array([6.4e-6] * 2)

    # Propagate point impulse
    resolution: Tuple = (1080, 1920)

    def setup(self):
        self.H = asm.compute(self.resolution, self.feature_size,
                             self.wavelength, self.z)

    @nn.compact
    def __call__(self, field):
        # Pad input field to the size of kernel
        print(self.H.shape)
        return field
        # H_h, H_w = self.H.shape
        # u_h, u_w = field.shape

        # pad_y, pad_x = jnp.array([H_h - u_h, H_w - u_w]) // 2
        # u_in = asm._pad(field, pad_y, pad_x)
        # return asm.propagate(u_in, self.H)


k1, k2 = random.split(random.PRNGKey(0), 2)
# grad(loss_fn)(variables)
point = jnp.zeros(Model.resolution, jnp.complex64)
m = Model()
variables = m.init(k1, point)


# loss_fn = lambda vars: jnp.linalg.norm(m.apply(vars, x))**2
@jit
def loss(variables, x):
    y = m.apply(variables, x)


embed()