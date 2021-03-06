from IPython import embed
from flax import linen as nn
from jax import numpy as jnp
from jax import random, grad


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


m = nn.ConvTranspose(features=10,
                     kernel_size=(2, 2),
                     dtype=jnp.complex64,
                     kernel_init=complex_kernel_init,
                     bias_init=complex_bias_init)
k1, k2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(k1, (10, 5, 5, 3))
variables = m.init(k1, x)

# loss_fn = lambda vars: jnp.linalg.norm(m.apply(vars, x))**2

y = m.apply(variables, x)

# grad(loss_fn)(variables)
embed()