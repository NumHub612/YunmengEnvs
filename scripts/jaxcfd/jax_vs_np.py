# -*- encoding: utf-8 -*-
"""
2d Diffusion equation solver using NumPy and jax.
"""
import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from numba import njit
import time


# variable declarations

nx = 1001
ny = 1001
nt = 5000
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((nx, ny))
un = np.ones((nx, ny))  # placeholder of solution

# initial condition
u[int(0.5 / dx) : int(1 / dx + 1), int(0.5 / dy) : int(1 / dy + 1)] = 2

# fig = plt.figure(figsize=(11, 7), dpi=100)
# ax = fig.add_subplot(111, projection="3d")
# X, Y = np.meshgrid(x, y)
# surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# plt.title("Initial Condition")
# plt.show()


# NumPy implementation


@njit
def diffusion_numpy(u, nt, dx, dy, nu, dt):

    # loop across number of time steps
    for n in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + nu * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            + nu * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
        )
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    return u


# Testing the NumPy implementation

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, nx)
X, Y = np.meshgrid(x, y)
## inital conditions
u_numpy = np.ones((nx, ny))

# set hat function I.C.: u(0.5<=x<=1 && 0.5<=y<=1) = 2
u_numpy[int(0.5 / dx) : int(1 / dx + 1), int(0.5 / dy) : int(1 / dy + 1)] = 2

# diffusion_numpy(u_numpy,nt,dx,dy,nu,dt)
t_start = time.time()
u_numpy = diffusion_numpy(u_numpy, nt, dx, dy, nu, dt)
t_end = time.time()
print("Execution time: ", t_end - t_start, "seconds")

# Plotting u_numpy
fig1 = plt.figure(figsize=(11, 7), dpi=100)
ax1 = fig1.add_subplot(111, projection="3d")
surf1 = ax1.plot_surface(X, Y, u_numpy, cmap=cm.viridis)
ax1.set_zlim(1, 2.5)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_title("u_numpy")
plt.show()


# Jax implementation
@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def stepper(u, nt, dx, dy, nu, dt):
    u = u.at[1:-1, 1:-1].set(
        (
            u[1:-1, 1:-1]
            + nu * dt / dx**2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])
            + nu * dt / dy**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
        )
    )
    # Bounday conditions
    u = u.at[0, :].set(1)
    u = u.at[-1, :].set(1)
    u = u.at[:, 0].set(1)
    u = u.at[:, -1].set(1)
    return u


def diffusion_jax(u, nt, dx, dy, nu, dt):

    for n in range(nt):
        u = stepper(u, nt, dx, dy, nu, dt)

    return u


# Testing the Jax implementation

u_jax = jnp.ones((nx, ny))  # create 2d array of 1's

# set hat function I.C.: u(0.5<=x<=1 && 0.5<=y<=1) = 2
u_jax = u_jax.at[int(0.5 / dx) : int(1 / dx + 1), int(0.5 / dy) : int(1 / dy + 1)].set(
    2
)

# The .at interface allows for a more familiar and compact notation,
# while preserving the immutable array semantics of JAX.
t_start = time.time()
u_jax = diffusion_jax(u_jax, nt, dx, dy, nu, dt)
t_end = time.time()
print("Execution time: ", t_end - t_start, "seconds")

# Plotting u_jax
fig2 = plt.figure(figsize=(11, 7), dpi=100)
ax2 = fig2.add_subplot(111, projection="3d")

surf2 = ax2.plot_surface(X, Y, u_jax, cmap=cm.viridis)
ax2.set_zlim(1, 2.5)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_title("u_jax")
plt.show()
