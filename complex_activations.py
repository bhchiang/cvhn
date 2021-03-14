from jax import numpy as jnp

def real_ReLU(x):
    return (x.real > 0) * x
    
def complex_ReLU(x):
    return (x.real > 0) * (x.imag > 0) * x
    
def complex_Cardiod(x):
    return  0.5 * (1 + x.real / (jnp.abs(x))) * x
    
def mod_ReLU(x, b=-1):
    return jnp.clip(jnp.abs(x)+b, a_min=0.0) / (jnp.abs(x)) * x
