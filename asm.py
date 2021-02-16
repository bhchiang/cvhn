import matplotlib.pyplot as plt
from jax import lax
from jax import numpy as jnp
from jax import vmap


def _pad(field, pad_widths):
    """Pad input field by widths.

    Args:
        field: jnp.ndarray - Input field to be padded.
        pad_widths: Iterable - Widths to pad each axis of field (rank 1). Length of field should
            be equal to the length of field.shape.
    """
    pad_widths = jnp.array(pad_widths)
    padding = jnp.stack((pad_widths, pad_widths)).T
    return jnp.pad(field, padding)


def propagate(u_in, H):
    """Propagates a single input field using the angular spectrum method.

    Args:
        u_in: jnp.ndarray, jnp.complex64 - Input complex tensor of size
            (height, width).
        H: jnp.ndarray, jnp.complex64 - Kernel in frequency domain containing phase shifts
            of size (H_height, H_width), where H_height > height and H_width > width.

    Returns:
        u_out: jnp.ndarray, jnp.complex64 - Propagated complex tensor of size
            (height, width).
    """

    # Pad input field to the size of kernel
    H_h, H_w = H.shape
    u_h, u_w = u_in.shape

    pad_widths = jnp.array([H_h - u_h, H_w - u_w]) // 2
    pad_h, pad_w = pad_widths
    u_in = _pad(u_in, pad_widths)

    # Decompose into angular spectrum of plane waves
    # norm="ortho" not supported yet - https://github.com/google/jax/issues/1877
    U1 = jnp.fft.fftn(jnp.fft.ifftshift(u_in))

    # Perform convolution
    U2 = H * U1

    u_out = jnp.fft.fftshift(jnp.fft.ifftn(U2))
    cropped = lax.slice(u_out, pad_widths, (H_h - pad_h, H_w - pad_w))
    # print(u_out.shape, cropped.shape)
    return cropped


def compute(u_in, feature_size, wavelength, z, kernel_size=-1):
    """Compute kernel of propagation terms to be used in the angular spectrum
        method.

    Args:
        u_in: jnp.ndarray, jnp.complex64 - Input complex tensor of size
            (height, width).
        feature_size: tuple - Tuple (height, width) of individual holographic
            features in meters.
        wavelength: float - Wavelength in meters.
        z: float - Propagation distance.
        kernel_size: float - Size of kernel in primal domain used to determine
            padding, -1 if kernel and scene are the same size.

    Returns:
        H: jnp.ndarray, jnp.complex64 - Complex kernel in the frequency domain
            containing phase shifts to multiply with input field.

    """
    # Compute padding
    input_resolution = u_in.shape
    pad_widths = jnp.array(
        [i // 2 if kernel_size == -1 else kernel_size for i in input_resolution]
    )
    u_in = _pad(u_in, pad_widths)
    # print(input_resolution, pad_widths, u_in.shape)

    # Compute kernel
    field_resolution = u_in.shape
    ny, nx = field_resolution
    dy, dx = feature_size
    y, x = (dy * float(ny), dx * float(nx))

    # Frequency coordinates sampling
    fy = jnp.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), ny)
    fx = jnp.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), nx)
    FX, FY = jnp.meshgrid(fx, fy)

    # Transfer function
    HH = (2 * jnp.pi) * jnp.sqrt(1 / (wavelength ** 2) - (FX ** 2 + FY ** 2))

    # Multiply by distance to get the final phase shift
    H_ = HH * z

    # Band-limited ASM - Matsushima et. al (2009)
    fy_max = 1 / jnp.sqrt((2 * jnp.abs(z) * (1 / y)) ** 2 + 1) / wavelength
    fx_max = 1 / jnp.sqrt((2 * jnp.abs(z) * (1 / x)) ** 2 + 1) / wavelength

    # Create mask
    H_f = jnp.uint8((jnp.abs(FX) < fx_max) & (jnp.abs(FY) < fy_max))

    # Convert to rectangular
    H_r = H_f * jnp.cos(H_)
    H_i = H_f * jnp.sin(H_)

    # Combine into complex
    H = H_r + H_i * 1j
    return jnp.fft.ifftshift(H)


if __name__ == "__main__":
    z = 0.05
    wavelength = 520e-9
    feature_size = [6.4e-6] * 2

    # Propagate point impulse
    h, w = (1080, 1920)
    point = jnp.zeros((h, w), dtype=jnp.complex64).at[h // 2, w // 2].set(1)
    H = compute(point, feature_size, wavelength, z)

    # plt.imshow(H.real, cmap="gray")
    # plt.show()

    # plt.imshow(H.imag, cmap="gray")
    # plt.show()

    propagated = propagate(point, H)

    fig, ax = plt.subplots(2, 1, figsize=(40, 20))
    ax[0].imshow(jnp.abs(propagated))
    ax[0].set_title("Magnitude")

    ax[1].imshow(propagated.real)
    ax[1].set_title("Real Component")

    plt.show()
    fig.savefig("images/propagated_point_visualization.png", bbox_inches="tight")
