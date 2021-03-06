from flax import linen as nn
from typing import Tuple
from jax import random, numpy as jnp
from IPython import embed


class Module(nn.Module):
    features: Tuple[int] = (16, 4)
    submodule: nn.Module = None

    def setup(self):
        self.dense1 = nn.Dense(self.features[0])
        self.dense2 = nn.Dense(self.features[1])

    @nn.compact
    def __call__(self, x):
        # Flatten input before using Linear layer
        print("Calling conv", x)
        x = nn.Conv(features=10, kernel_size=5, padding="VALID")(x)
        if self.submodule is not None:
            x = self.submodule(x)
        return x
        # return self.dense2(nn.relu(self.dense1(x)))


key = random.PRNGKey(0)
k1, k2 = random.split(key, 2)
d = random.uniform(k1, (20, 20, 3))
_model = Module()
model = Module(submodule=_model)

params = model.init(k2, d)
# y = model.apply(parcams, d)

embed()