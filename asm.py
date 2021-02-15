from jax import numpy as jnp, vmap


def propagate(u_in, H):
    """Propagates a single input field using the angular spectrum method.

    Args:
        u_in: jnp.ndarray, jnp.complex64 - Input complex tensor of size
            (height, width).

    Returns:
        u_out: jnp.ndarray, jnp.complex64 - Propagated complex tensor of size
            (height, width).

    """
    pass


def compute(u_in, feature_size, wavelength, z, kernel_size=-1):
    """Compute kernel of propagation terms to be used in the angular spectrum method.

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
    pass


if __name__ == "__main__":
    # Test functions (put somewhere else later)
    pass
