import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imports import *

def plotty(adimensionalSolution, phaseID, param, OBoptimalTrajectory):

	initialStateTarget_S = adimensionalSolution['y'][0, 0:6]
	initialStateChaser_S = adimensionalSolution['y'][0, 6:12]
	DeltaIC_S = initialStateChaser_S - initialStateTarget_S

	# PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
	sol = adimensionalSolution['y'].T
	chaserPosition = sol[:, 6:9]
	targetPosition = sol[:, 0:3]
	xc = param.xc

	relDynami = np.zeros((len(adimensionalSolution['x']), 6))

	for id in range(len(adimensionalSolution['x'])):
		rotatedRelativeState = ReferenceFrames.convert_S_to_LVLH(sol[id, 0:6], sol[id, 6:12] - sol[id, 0:6], param)
		relDynami[id, 0:6] = rotatedRelativeState

	controlAction = adimensionalSolution['controlAction']
	relDynami *= xc

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if OBoptimalTrajectory is not None and "x" in OBoptimalTrajectory:
		OBOT = OBoptimalTrajectory['x'].T * xc
		ax.plot(OBOT[:, 0], OBOT[:, 1], OBOT[:, 2], 'm--', linewidth=1.2, label="Optimal Trajectory")

	ax.plot(relDynami[:, 0], relDynami[:, 1], relDynami[:, 2], 'b', linewidth=1.2, label="Actual Trajectory")
	ax.plot(0, 0, 0, 'r*', linewidth=2)

	DeltaIC_L = ReferenceFrames.convert_S_to_LVLH(initialStateTarget_S, DeltaIC_S, param)
	DeltaICm = DeltaIC_L[0:3] * xc
	ax.quiver(0, 0, 0, relDynami[0, 0], relDynami[0, 1], relDynami[0, 2], color='k', linewidth=0.9)
	ax.plot(relDynami[0, 0], relDynami[0, 1], relDynami[0, 2], 'ok', linewidth=1)

	ax.quiver(relDynami[:, 0], relDynami[:, 1], relDynami[:, 2], controlAction[0, :], controlAction[1, :], controlAction[2, :], color='g', linewidth=0.8, label="Control Action")

	ax.quiver(0, 0, 0, np.linalg.norm(DeltaICm), 0, 0, color='r', linewidth=1)
	ax.quiver(0, 0, 0, 0, np.linalg.norm(DeltaICm), 0, color='r', linewidth=1)
	ax.quiver(0, 0, 0, 0, 0, np.linalg.norm(DeltaICm), color='r', linewidth=1)

	holdState = param.holdingState * xc
	ax.plot(holdState[0], holdState[1], holdState[2], 'db', linewidth=1, label="Hold Position")

	if phaseID == 1:
		plotConstraintsVisualization(1e3, 'S', 'yellow')
		plotConstraintsVisualization(200, 'S')
		plotConstraintsVisualization(2.5e3, 'S', '#808080')
	elif phaseID == 2:
		plotConstraintsVisualization(1e3, 'C')

	ax.legend(loc='best')
	ax.set_xlabel("R-bar [km]")
	ax.set_ylabel("V-bar [km]")
	ax.set_zlabel("H-bar [km]")
	ax.set_title("Relative Dynamics [CONTROLLED]")
	ax.grid(True)
	ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

	# Zoom in for phase 2
	if phaseID == 2:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		if OBoptimalTrajectory is not None and "x" in OBoptimalTrajectory:
			OBOT = OBoptimalTrajectory['x'].T * xc
			ax.plot(OBOT[:, 0], OBOT[:, 1], OBOT[:, 2], 'm--', linewidth=1.2, label="Optimal Trajectory")

		ax.plot(relDynami[:, 0], relDynami[:, 1], relDynami[:, 2], 'b', linewidth=1.2, label="Actual Trajectory")
		ax.plot(0, 0, 0, 'r*', linewidth=2)

		ax.quiver(0, 0, 0, relDynami[0, 0], relDynami[0, 1], relDynami[0, 2], color='k', linewidth=0.9)
		ax.plot(relDynami[0, 0], relDynami[0, 1], relDynami[0, 2], 'ok', linewidth=1)

		ax.quiver(relDynami[:, 0], relDynami[:, 1], relDynami[:, 2], controlAction[0, :], controlAction[1, :], controlAction[2, :], color='g', linewidth=0.8, label="Control Action")

		ax.quiver(0, 0, 0, np.linalg.norm(DeltaICm), 0, 0, color='r', linewidth=1)
		ax.quiver(0, 0, 0, 0, np.linalg.norm(DeltaICm), 0, color='r', linewidth=1)
		ax.quiver(0, 0, 0, 0, 0, np.linalg.norm(DeltaICm), color='r', linewidth=1)

		plotConstraintsVisualization(200, 'C')

		ax.legend(loc='best')
		ax.set_xlabel("R-bar [km]")
		ax.set_ylabel("V-bar [km]")
		ax.set_zlabel("H-bar [km]")
		ax.set_title("Relative Dynamics [CONTROLLED]")
		ax.grid(True)
		ax.set_xlim([-0.1, 0.1])
		ax.set_ylim([-0.1, 0])
		ax.set_zlim([-0.1, 0.1])

	# Plot control actions and relative dynamics
	t = adimensionalSolution['x'] * param.tc / 60
	fig, axs = plt.subplots(3, 1, figsize=(8, 12))

	axs[0].plot(t, controlAction.T, linewidth=1.1)
	axs[0].grid(True)
	axs[0].set_title("Control Action [LVLH]")
	axs[0].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[0].set_xlabel("Time [min]")
	axs[0].set_ylabel("Control Action [-]")

	axs[1].plot(t, relDynami[:, 0:3], linewidth=1)
	axs[1].set_title("Controlled Relative Dynamics [LVLH]")
	axs[1].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[1].grid(True)
	axs[1].set_xlabel("Time [min]")
	axs[1].set_ylabel("Position [km]")

	axs[2].plot(t, relDynami[:, 3:6] / param.tc, linewidth=1)
	axs[2].set_title("Controlled Relative Velocity [LVLH]")
	axs[2].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[2].grid(True)
	axs[2].set_xlabel("Time [min]")
	axs[2].set_ylabel("Velocity [km/s]")

	plt.tight_layout()
	plt.show()




def plotConstraintsVisualization(DeltaIC_meters, type='C', colore=None):
	# coefficients definition
	acone = 0.08
	bcone = 5

	if DeltaIC_meters < 200:
		rsphere = 200  # [m]
	else:
		rsphere = DeltaIC_meters

	if type.upper() == 'S':
		plot_sphere(rsphere, colore)
	elif type.upper() == 'C':
		plot_cone(DeltaIC_meters, colore, acone, bcone)

	plt.xlabel("R-BAR")
	plt.ylabel("V-BAR")
	plt.gca().set_zlabel("H-BAR")
	plt.axis('equal')
	plt.show()

def plot_sphere(rsphere, colore):
	z = lambda RbarX, VbarX: np.real(np.sqrt(rsphere**2 - RbarX**2 - VbarX**2)) * 1e-3
	pointsR = np.linspace(-rsphere, rsphere, 501)
	pointsV = np.linspace(-rsphere, rsphere, 501)
	X, Y = np.meshgrid(pointsR, pointsV)

	sferaz = z(X, Y)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if X[i, j]**2 + Y[i, j]**2 > (rsphere * 1.01)**2:
				sferaz[i, j] = np.nan

	colore = colore if colore else 'red'
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X * 1e-3, Y * 1e-3, sferaz, color=colore, alpha=0.5, edgecolor='none')
	ax.plot_surface(X * 1e-3, Y * 1e-3, -sferaz, color=colore, alpha=0.5, edgecolor='none')

def plot_cone(DeltaIC_meters, colore, acone, bcone):
	z = lambda RbarX, VbarX: np.real(np.sqrt(acone**2 * bcone**3 - 3 * acone**2 * bcone**2 * VbarX + 3 * acone**2 * bcone * VbarX**2 - acone**2 * VbarX**3 - RbarX**2))
	pointsR = np.linspace(-3 * DeltaIC_meters, 3 * DeltaIC_meters, 501)
	pointsV = np.linspace(-DeltaIC_meters, 0, 51)
	X, Y = np.meshgrid(pointsR, pointsV)

	halfCone = z(X, Y)  # [m]
	toll = max(np.max(np.diff(pointsR)), np.max(np.diff(pointsV))) * 1.9
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if -(Y[i, j] - toll - bcone)**3 < (X[i, j]**2 / acone**2):
				halfCone[i, j] = np.nan

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X * 1e-3, Y * 1e-3, halfCone * 1e-3, color='black', alpha=0.5, edgecolor='none')
	ax.plot_surface(X * 1e-3, Y * 1e-3, -halfCone * 1e-3, color='black', alpha=0.5, edgecolor='none')
