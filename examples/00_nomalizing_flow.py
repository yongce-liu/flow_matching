# %% [markdown]
# ## Blog
# - https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
# - Discrete Normalising Flows
# %%
from collections.abc import Callable
from typing import TypeAlias

import jax
from jax import Array
from jax import scipy as jsp
import jax.numpy as jnp
from jaxtyping import Float
import matplotlib.pyplot as plt

RNDKey = jax.random.PRNGKey(0)

# %%

TInput: TypeAlias = Float[Array, "dim"]
TOutput: TypeAlias = Float[Array, ""]


def est_p1(
    x: TInput,
    phi: Callable[[TInput], TOutput],
    p0: Callable[[TInput], TOutput],
) -> TOutput:
    # Use vmap to compute determinants for the batch efficiently
    # Calculate Jacobian of phi_inv w.r.t a single sample y
    j = jax.jacfwd(phi)(x)
    # Calculate absolute determinant
    return p0(x) / jnp.abs(jnp.linalg.det(j))


# %%
# an analytical sol
dim = 1
mu = jnp.zeros((dim,))
cov = jnp.eye(dim)
a, b = 2, 1
p0 = lambda x, loc=mu, scale=cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)
phi = lambda x, a=a, b=b: a * x + b
p1 = lambda x, loc=a * mu + b, scale=a**2 * cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)

# Compute mean absolute error
samples = 10 * (jax.random.uniform(RNDKey, (1000, dim)) - 0.5)
val_p0 = jax.vmap(p0)(samples)
val_estp1 = jax.vmap(lambda x: est_p1(x, phi, p0))(samples)
samples_y = jax.vmap(phi)(samples)
val_p1 = jax.vmap(p1)(samples_y)


# %%
loss = jnp.abs(val_estp1 - val_p1).mean()
print(f"Loss (Mean deviation): {loss}")
# Create grid for plotting
plt.figure(figsize=(10, 5))
plt.scatter(samples, val_p0, label="Ori p(x)")
plt.scatter(samples_y, val_p1, label="True q(y)")
plt.scatter(samples_y, val_estp1, label="Estimated q(y)", marker="x", s=1)
plt.legend()
plt.show()

# %%
dim = 1
mu = jnp.zeros((dim,))
cov = jnp.eye(dim)
p0 = lambda x, loc=mu, scale=cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)
phi = lambda x: jnp.sin(x)

# Compute mean absolute error
samples = 2 * (jax.random.uniform(RNDKey, (1000, dim)) - 0.5)
val_p0 = jax.vmap(p0)(samples)
val_estp1 = jax.vmap(lambda x: est_p1(x, phi, p0))(samples)
samples_y = jax.vmap(phi)(samples)

# %%
# Create grid for plotting
plt.figure(figsize=(10, 5))
plt.scatter(samples, val_p0, label="Ori p(x)")
plt.scatter(samples_y, val_estp1, label="Estimated q(y)", marker="x", s=1)
plt.legend()
plt.show()

# %%
