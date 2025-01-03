import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generalScripts.OBGuidance import ASRE
from generalScripts.OBNavigation import OBNavigation

from generalScripts.ReferenceFrames import rotate_M_to_LVLH
import config

# Initialize simulation
param, initialValue = config.env_config.get()

initial_state_target_M, initial_state_chaser_M, initial_relative_state_L = OBNavigation(
    initialValue.targetState_S, initialValue.chaserState_S, param
)
TOF = 0.0333
optimal_trajectory = ASRE(TOF, initial_relative_state_L, initial_state_target_M, param)

# Plotting relative dynamics inside LVLH frame
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(
    optimal_trajectory["x"][0, :] * param.xc,
    optimal_trajectory["x"]	[1, :] * param.xc,
    optimal_trajectory["x"]	[2, :] * param.xc,
    'b.-', linewidth=1.1
)
ax.scatter(
    optimal_trajectory["x"]	[0, 0] * param.xc,
    optimal_trajectory["x"]	[1, 0] * param.xc,
    optimal_trajectory["x"]	[2, 0] * param.xc,
    c='k', label='Initial Chaser Position'
)
ax.scatter(
    optimal_trajectory["x"]	[0, -1] * param.xc,
    optimal_trajectory["x"]	[1, -1] * param.xc,
    optimal_trajectory["x"]	[2, -1] * param.xc,
    c='k', marker='*', label='Final Chaser Position'
)
ax.scatter(0, 0, 0, c='r', marker='*', s=100, label='Target Position')

DeltaIC_L = rotate_M_to_LVLH(initial_state_target_M, initial_state_chaser_M - initial_state_target_M, param)
DeltaICm = DeltaIC_L[:3] * param.xc

ax.quiver(0, 0, 0, DeltaICm[0], 0, 0, color='r', linewidth=1)
ax.quiver(0, 0, 0, 0, DeltaICm[1], 0, color='r', linewidth=1)
ax.quiver(0, 0, 0, 0, 0, DeltaICm[2], color='r', linewidth=1)

ax.legend(loc='best')
ax.set_title("ONBOARD-COMPUTED Relative Dynamics in Target LVLH")
ax.set_xlabel("R-bar [km]")
ax.set_ylabel("V-bar [km]")
ax.set_zlabel("H-bar [km]")
ax.grid(True)
plt.show()

# Plot trajectory components
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(optimal_trajectory["t"]	 * param.tc, optimal_trajectory["x"]	[0, :] * param.xc, linewidth=1)
axs[0].set_title("ONBOARD-COMPUTED Trajectory on R-bar")
axs[0].grid(True)

axs[1].plot(optimal_trajectory["t"]	 * param.tc, optimal_trajectory["x"]	[1, :] * param.xc, linewidth=1)
axs[1].set_title("ONBOARD-COMPUTED Trajectory on V-bar")
axs[1].grid(True)

axs[2].plot(optimal_trajectory["t"]	 * param.tc, optimal_trajectory["x"]	[2, :] * param.xc, linewidth=1)
axs[2].set_title("ONBOARD-COMPUTED Trajectory on H-bar")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Plot control actions
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(optimal_trajectory["t"]	 * param.tc / 60, optimal_trajectory["u"][0, :] / param.tc**2 * param.xc * 1e3, linewidth=1)
axs[0].set_title("Control Action on R-bar")
axs[0].grid(True)

axs[1].plot(optimal_trajectory["t"]	 * param.tc / 60, optimal_trajectory["u"]	[1, :] / param.tc**2 * param.xc * 1e3, linewidth=1)
axs[1].set_title("Control Action on V-bar")
axs[1].grid(True)

axs[2].plot(optimal_trajectory["t"]	 * param.tc / 60, optimal_trajectory["u"]	[2, :] / param.tc**2 * param.xc * 1e3, linewidth=1)
axs[2].set_title("Control Action on H-bar")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Plot controlled dynamics
rel_dynamics = optimal_trajectory["x"].T * param.xc
control_actions = optimal_trajectory["u"].T

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(optimal_trajectory["t"]	 * param.tc / 60, control_actions, linewidth=1.1)
axs[0].set_title("Control Action")
axs[0].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
axs[0].grid(True)

axs[1].plot(optimal_trajectory["t"]	 * param.tc / 60, rel_dynamics[:, :3], linewidth=1)
axs[1].set_title("Controlled Relative Dynamics [LVLH]")
axs[1].grid(True)

axs[2].plot(optimal_trajectory["t"]	 * param.tc / 60, rel_dynamics[:, 3:] / param.tc, linewidth=1)
axs[2].set_title("Controlled Relative Velocity [LVLH]")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Final 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(rel_dynamics[:, 0], rel_dynamics[:, 1], rel_dynamics[:, 2], 'b', linewidth=1.2)
ax.scatter(0, 0, 0, c='r', marker='*', s=100, label='Target LVLH')
ax.quiver(
    0, 0, 0,
    rel_dynamics[0, 0], rel_dynamics[0, 1], rel_dynamics[0, 2],
    color='k', linewidth=0.9
)
ax.quiver(
    rel_dynamics[:, 0], rel_dynamics[:, 1], rel_dynamics[:, 2],
    control_actions[:, 0], control_actions[:, 1], control_actions[:, 2],
    color='g', linewidth=0.8
)
ax.legend(loc='best')
ax.set_title("Actual Relative Dynamics [WITH CONTROL SHOWN]")
ax.set_xlabel("R-bar [km]")
ax.set_ylabel("V-bar [km]")
ax.set_zlabel("H-bar [km]")
ax.grid(True)
plt.show()