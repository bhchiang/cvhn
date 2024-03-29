import matplotlib.pyplot as plt
from jax import lax
from jax import numpy as jnp
import jax


def _pad(field, pad_y, pad_x):
    """Pad input field by widths.

    Args:
        field: jnp.ndarray - Input field to be padded of shape (height, width).
        pad_sizes: jnp.ndarray[int, int] - Widths to pad each axis of field (rank 1). Length of field should
            be equal to the length of field.shape.

    Returns:
        padded: jnp.ndarray - Padded field of shape = field.shape + pad_sizes * 2.
    """
    return jnp.pad(field, ((pad_y, pad_y), (pad_x, pad_x)))


def _crop(field, pad_y, pad_x):
    """Crop input field by pad sizes.
    Args:
        field: jnp.ndarray - Input field to be cropped of shape (height, width).
        pad_sizes: jnp.ndarray[int, int] - Width of pads to remove from each side of 
            each axis of the field.

    Returns:            
        cropped: jnp.ndarray - Cropped field of shape = field.shape - pad_sizes * 2.
    """
    h, w = field.shape
    return lax.slice(field, (pad_y, pad_x), (h - pad_y, w - pad_x))


@jax.jit
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

    # # Pad input field to the size of kernel
    # H_h, H_w = H.shape
    # u_h, u_w = u_in.shape

    # pad_y, pad_x = jnp.array([H_h - u_h, H_w - u_w]) // 2
    # u_in = _pad(u_in, pad_y, pad_x)

    # Decompose into angular spectrum of plane waves
    # norm="ortho" not supported yet - https://github.com/google/jax/issues/1877
    U1 = jnp.fft.fftn(jnp.fft.ifftshift(u_in))

    # Perform convolution
    U2 = H * U1

    u_out = jnp.fft.fftshift(jnp.fft.ifftn(U2))
    # cropped = _crop(u_out, pad_y, pad_x)
    return u_out


def compute(input_resolution, feature_size, wavelength, d, kernel_size=-1):
    """Compute kernel of propagation terms to be used in the angular spectrum
        method.
    Args:
        input_resolution: Tuple (height, width) indicating size of  input complex tensor.
        feature_size: tuple - Tuple (height, width) of individual holographic
            features in meters.
        wavelength: float - Wavelength in meters.
        d: float - Propagation distance.
        kernel_size: float - Size of kernel in primal domain used to determine
            padding, -1 if kernel and scene are the same size.
    Returns:
        H: jnp.ndarray, jnp.complex64 - Complex kernel in the frequency domain
            containing phase shifts to multiply with input field.
    """
    # Compute padding
    pad_widths = jnp.array([
        s // 2 if kernel_size == -1 else kernel_size for s in input_resolution
    ])
    field_resolution = jnp.array(
        [x + y for x, y in zip(input_resolution, pad_widths)])

    # Compute kernel
    ny, nx = field_resolution
    dy, dx = feature_size
    y, x = (dy * jnp.float32(ny), dx * jnp.float32(nx))

    # Frequency coordinates sampling
    fy = jnp.linspace(-1 / (2 * dy) + 0.5 / (2 * y),
                      1 / (2 * dy) - 0.5 / (2 * y), ny)
    fx = jnp.linspace(-1 / (2 * dx) + 0.5 / (2 * x),
                      1 / (2 * dx) - 0.5 / (2 * x), nx)
    # print(fx[:10])
    # print(fy[:10])
    FX, FY = jnp.meshgrid(fx, fy)

    # Transfer function
    HH = (2 * jnp.pi) * jnp.sqrt(1 / (wavelength**2) - (FX**2 + FY**2))

    # Multiply by distance to get the final phase shift
    H_ = HH * d

    # Band-limited ASM - Matsushima et. al (2009)
    fy_max = 1 / jnp.sqrt((2 * jnp.abs(d) * (1 / y))**2 + 1) / wavelength
    fx_max = 1 / jnp.sqrt((2 * jnp.abs(d) * (1 / x))**2 + 1) / wavelength

    # Create mask
    H_f = jnp.uint8((jnp.abs(FX) < fx_max) & (jnp.abs(FY) < fy_max))

    # Convert to complex
    H = H_f * jnp.exp(H_ * 1j)
    return jnp.fft.ifftshift(H)


if __name__ == "__main__":
    d = 0.05  # Propagation distance
    wavelength = 520e-9
    feature_size = jnp.array([6.4e-6] * 2)

    # Propagate point impulse
    h, w = (1080, 1920)
    point = jnp.zeros((h, w), dtype=jnp.complex64).at[h // 2, w // 2].set(1)
    H = compute(point.shape, feature_size, wavelength, d)

    # plt.imshow(H.real)
    # plt.show()

    # plt.imshow(H.imag)
    # plt.show()

    propagated = propagate(point, H)

    plt.imsave("images/propagated_magnitude.png", jnp.abs(propagated))
    plt.imsave("images/propagated_real.png", propagated.real)
    plt.imsave("images/propagated_imaginary.png", propagated.imag)

    # Propagate the propagated impulse back by -z and verify we get the point back
    propagated_back = jnp.conj(propagate(jnp.conj(propagated), H))

    def center(field, size=50):
        return lax.slice(
            field,
            (h // 2 - size // 2, w // 2 - size // 2),
            (h // 2 + size // 2, w // 2 + size // 2),
        )

    plt.imsave(
        "images/propagated_back_center.png",
        jnp.abs(center(propagated_back)),
    )
    plt.imsave("images/point_center.png", jnp.abs(center(point)))

    def _mse(a, b):
        return jnp.mean(jnp.abs(a - b)**2)

    mse = _mse(point, propagated_back)
    print(f"MSE = {mse}")
