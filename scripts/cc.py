from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from IPython import embed

# Complex convolutions in JAX
# Create 2D kernel - HWIO layout
kernel = jnp.zeros((3, 3, 3, 3), dtype=jnp.float32)
kernel += jnp.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])[
    :, :, jnp.newaxis, jnp.newaxis
]

plt.figure()
plt.imshow(kernel[:, :, 0, 0])
plt.title("Edge Convolution Kernel")
plt.show()

# NHWC layout
img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
for k in range(3):
    x = 30 + 60 * k
    y = 20 + 60 * k
    img = img.at[0, x : x + 10, y : y + 10, k].set(1.0)

plt.title("Original Image")
plt.imshow(img[0])
plt.show()


# Image should be NCHW
# Kernel should be OIHW
# I = # of input channels = # of channels in image
out = lax.conv(
    jnp.transpose(img, [0, 3, 1, 2]).astype(jnp.complex64),
    jnp.transpose(kernel, [3, 2, 0, 1]).astype(jnp.complex64),
    (1, 1),  # window strides
    "SAME",
)  # padding mode

print(out.shape)
embed()
plt.figure(figsize=(10, 10))
plt.imshow(jnp.array(out)[0, 0, :, :])
plt.show()
