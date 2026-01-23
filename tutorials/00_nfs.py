# %% [markdown]
#  # Flow Matching

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

# %% [markdown]
#  ## Change of Variable for Probability Density Function
#
#  - $x_0 \sim p_0(x_0)$
#
#  - $p_1(x_1) = p_0(x_0)\cdot {|\det \frac{\partial \phi(x_0)}{\partial x_0}|}^{-1}$, where $x_1 = \phi(x_0)$

# %% [markdown]
# ### With a know transformation $x_1=\phi(x_0)$

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
    jac = jax.jacfwd(phi)(x)
    # Calculate absolute determinant
    return p0(x) / jnp.abs(jnp.linalg.det(jac))


# %%
dim = 1
mu = jnp.zeros((dim,))
cov = jnp.eye(dim)
a, b = 2, 1
p0 = lambda x, loc=mu, scale=cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)  # noqa: E731
phi = lambda x, a=a, b=b: a * x + b  # noqa: E731
p1 = lambda x, loc=a * mu + b, scale=a**2 * cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)  # noqa: E731

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


# %%
dim = 1
mu = jnp.zeros((dim,))
cov = jnp.eye(dim)
p0 = lambda x, loc=mu, scale=cov: jsp.stats.multivariate_normal.pdf(x, loc, scale)  # noqa: E731
phi = lambda x: jnp.sin(x)  # noqa: E731

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


# %% [markdown]
# ### With the data $p_0(x_0)$ and label $p_1(x_1)$. Leaning the bi-jective transformation $\phi$ to match $x_0$ to $x_1$.


# %%
# 1. Target Distribution (Data)
# We treat p1(x1) as the 'data' distribution we want to learn.
# For a robust "simple" flow (Affine), we target a Gaussian N(mu=4, sigma=2).
target_mu, target_scale = 4.0, 2.0
data_dim = 1
n_samples = 2000

# Use a new key derived from RNDKey and some offset to ensure independence
key_train = jax.random.fold_in(RNDKey, 999)
key_data, key_init = jax.random.split(key_train)

# Generate observations from p1
x0_samples = jax.random.normal(key_data, (n_samples, data_dim))
x1_samples = target_mu + target_scale * x0_samples

# %%
# 2. Define Learning Model (Affine Flow)
# We want to learn a bijection phi: x0 -> x1 such that x1 ~ p1 when x0 ~ p0.
# The normalizing direction is x1 -> x0 (Data -> Base).
# x0 = (x1 - loc) / scale  =>  x1 = x0 * scale + loc
# We maximize the likelihood of the data p1_samples under this model.


def init_params(k):
    # Learnable parameters: location and log_scale
    return {"loc": jnp.zeros((data_dim,)), "log_scale": jnp.zeros((data_dim,))}


def inverse_flow(params, x1):
    """Normalizing direction: x1 (Data) -> x0 (Base). Returns x0 and log|det J|."""
    scale = jnp.exp(params["log_scale"])
    x0 = (x1 - params["loc"]) / scale
    # Jacobian d(x0)/d(x1) = 1/scale. log_det = -log(scale) = -log_scale
    log_det = -params["log_scale"].sum()
    return x0, log_det


def forward_flow(params, x0):
    """Generative direction: x0 (Base) -> x1 (Data)."""
    scale = jnp.exp(params["log_scale"])
    x1 = x0 * scale + params["loc"]
    return x1


# %%
# 3. Loss Function & Training
# Maximize Log Likelihood <=> Minimize Negative Log Likelihood (NLL)


def p0_log_prob(x):
    # Base distribution: Standard Normal N(0, 1)
    return jsp.stats.norm.logpdf(x).sum(axis=-1)


def loss_fn(params, x1):
    x0, log_det = inverse_flow(params, x1)
    # log p_model(x1) = log p0(x0) + log |det J|
    log_lik_p1 = p0_log_prob(x0) + log_det
    return -jnp.mean(log_lik_p1)


# Optimization Setup
learning_rate = 0.1
params = init_params(key_init)
grad_fn = jax.grad(loss_fn)

loss_history = []
print("Start Training...")

for step in range(500):
    grads = grad_fn(params, x1_samples)
    # Manual Gradient Descent Update
    params["loc"] = params["loc"] - learning_rate * grads["loc"]
    params["log_scale"] = params["log_scale"] - learning_rate * grads["log_scale"]

    if step % 100 == 0:
        loss = loss_fn(params, x1_samples)
        loss_history.append(loss)
        print(f"Step {step:03d} | Loss: {loss:.4f}")

print(f"Learned: loc={params['loc'][0]:.4f}, scale={jnp.exp(params['log_scale'][0]):.4f}")
print(f"Target:  loc={target_mu:.4f}, scale={target_scale:.4f}")

# %%
# 4. Visualization
plt.figure(figsize=(10, 4))

# Plot 1: Compare Data Density vs Model Density
ax1 = plt.subplot(1, 2, 1)
# Generate samples from the learned model
x1_model = forward_flow(params, x0_samples)

ax1.hist(x1_samples.flatten(), bins=50, density=True, alpha=0.5, label="Target Data p1")
ax1.hist(x1_model.flatten(), bins=50, density=True, alpha=0.5, label="Learned Model")
ax1.set_title("Data Space (x1)")
ax1.legend()

# %%
# Plot 2: Check Latent Space (Should be Standard Normal)
ax2 = plt.subplot(1, 2, 2)
x0_mapped, _ = inverse_flow(params, x1_samples)
grid = jnp.linspace(-4, 4, 100)

ax2.hist(x0_mapped.flatten(), bins=50, density=True, alpha=0.5, label="Mapped Data x0")
ax2.plot(grid, jsp.stats.norm.pdf(grid), "k--", label="Prior N(0,1)")
ax2.set_title("Latent Space (x0)")
ax2.legend()

plt.tight_layout()
plt.show()

# %% multiple nfs
# ...
