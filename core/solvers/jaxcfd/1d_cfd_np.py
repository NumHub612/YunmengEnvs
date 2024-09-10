# -*- encoding: utf-8 -*-
"""
1d Burgers equation solver using NumPy and Numba.
"""
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from matplotlib import pyplot as plt


# set up symbolic variables for the three variables in our initial condition
x, nu, t = sp.symbols("x nu t")
phi = sp.exp(-((x - 4 * t) ** 2) / (4 * nu * (t + 1))) + sp.exp(
    -((x - 4 * t - 2 * sp.pi) ** 2) / (4 * nu * (t + 1))
)
print(phi)

# evaluate the partial derivative
phiprime = phi.diff(x)

# writing out the full initial condition equation
u = -2 * nu * (phiprime / phi) + 4

# lambdify this expression into a useable function
u_func = lambdify((t, x, nu), u)
val = u_func(1, 4, 3)
print(val)


# variable declarations
nx = 401
nt = 100
dx = 2 * np.pi / (nx - 1)

nu = 0.07
dt = dx * nu

x = np.linspace(0, 2 * np.pi, nx)
un = np.zeros(nx)

# set initial condition
t = 0
u = np.asarray([u_func(t, x0, nu) for x0 in x])
# print(u)


# Set default plot settings
plt.rcParams["font.family"] = "arial"
plt.rcParams["font.size"] = 16

# # Create a new figure
# plt.figure(figsize=(11, 7), dpi=100)
# plt.plot(x, u, color="blue", linewidth=2, marker="o", label="Initial")
# plt.xlabel("x", fontsize=16)
# plt.ylabel("u0", fontsize=16)
# plt.title("Plot of the initial condition", fontsize=18)

# # Add a grid
# plt.grid(True)

# # Add a legend with a specific location
# plt.legend(loc="upper right")

# # Set the limit for the x and y axes
# plt.xlim([0, 2 * np.pi])
# plt.ylim([0, 10])

# # Set the ticks for the x and y axes
# plt.xticks(np.arange(0, 2 * np.pi, 1))
# plt.yticks(np.arange(0, 10, 1))

# # Display the plot
# plt.show()

# core proceeding.
for n in range(nt):
    un = u.copy()
    # update all inner points at once
    for i in range(1, nx - 1):
        u[i] = (
            un[i]
            - un[i] * dt / dx * (un[i] - un[i - 1])
            + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
        )
    # set boundary conditions
    u[-1] = (
        un[-1]
        - un[-1] * dt / dx * (un[-1] - un[-2])
        + nu * dt / dx**2 * (un[1] - 2 * un[-1] + un[-2])
    )
    u[0] = u[-1]

u_analy = np.asarray([u_func(nt * dt, xi, nu) for xi in x])

# Create a new figure
plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x, u, color="blue", linewidth=2, marker="o", label="Computational")
plt.plot(x, u_analy, color="red", linewidth=2, label="Analytical")
plt.xlabel("x", fontsize=16)
plt.ylabel("u", fontsize=16)
plt.title("Comparison of computational and analytical results", fontsize=18)

plt.grid(True)
plt.legend(loc="upper right")
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])

plt.xticks(np.arange(0, 2 * np.pi, 1))
plt.yticks(np.arange(0, 10, 1))
plt.show()
