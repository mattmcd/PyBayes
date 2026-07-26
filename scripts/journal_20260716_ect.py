# Gemini code

import numpy as np
import matplotlib.pyplot as plt

# 1. Setup discretization parameters
num_angles = 360  # Reasonable angular resolution (every 10 degrees)
num_heights = 1000  # Density of the sweep lines
theta_vals = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
t_vals = np.linspace(-1.5, 1.5, num_heights)

# Define the vertices of the central square (1x1 centered at 0)
vertices = np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5]
])

# Initialize the ECT data matrices
ect_solid = np.zeros((num_angles, num_heights))
ect_boundary = np.zeros((num_angles, num_heights))

# 2. Compute the ECT for each direction and height
for i, theta in enumerate(theta_vals):
    v = np.array([np.cos(theta), np.sin(theta)])

    # Project vertices onto the sweep direction vector to find critical heights
    proj_heights = np.dot(vertices, v)
    t_min, t_max = np.min(proj_heights), np.max(proj_heights)

    # Sort projections to identify transition zones
    sorted_proj = np.sort(proj_heights)
    t1, t2, t3, t4 = sorted_proj  # 4 corner contact heights

    for j, t in enumerate(t_vals):
        # --- Case 1: Solid Square ---
        if t < t_min:
            ect_solid[i, j] = 0
        else:
            ect_solid[i, j] = 1

        # --- Case 2: Hollow Boundary ---
        if t < t1:
            # Not yet reached
            ect_boundary[i, j] = 0
        elif t1 <= t < t2:
            # Touched the first corner: 1 connected component (a V-shape)
            ect_boundary[i, j] = 1
        elif t2 <= t < t3:
            # Sweeping through the middle: 2 disconnected segments
            ect_boundary[i, j] = 2
        elif t3 <= t < t4:
            # Reached the final corner: back to 1 connected component
            ect_boundary[i, j] = 1
        else:
            # Entire loop is swallowed: Homeomorphic to S^1
            ect_boundary[i, j] = 0

# 3. Visualize the ECT data matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot Solid Square ECT
im0 = axes[0].imshow(ect_solid, extent=[-1.5, 1.5, 0, 360], origin='lower',
                     aspect='auto', cmap='viridis', interpolation='nearest')
axes[0].set_title('ECT Matrix: Solid Square $A$')
axes[0].set_xlabel('Sweep Height ($t$)')
axes[0].set_ylabel('Sweep Direction ($\\theta$ in degrees)')
fig.colorbar(im0, ax=axes[0], label='Euler Characteristic $\\chi$')

# Plot Hollow Boundary ECT
im1 = axes[1].imshow(ect_boundary, extent=[-1.5, 1.5, 0, 360], origin='lower',
                     aspect='auto', cmap='plasma', interpolation='nearest')
axes[1].set_title('ECT Matrix: Hollow Boundary $A$')
axes[1].set_xlabel('Sweep Height ($t$)')
fig.colorbar(im1, ax=axes[1], label='Euler Characteristic $\\chi$')

plt.tight_layout()
plt.show()