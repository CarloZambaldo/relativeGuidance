import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generalScripts import *

#import seaborn as sns

def plotty(env):
	# sns.set_theme(style="whitegrid")

	adimensionalSolution = {
		"x": env.timeHistory,
		"y": env.fullStateHistory,
		"controlAction": env.controlActionHistory_L[:-1,:],
	}
	phaseID = env.param.phaseID
	param = env.param
	OBoptimalTrajectory = env.OBoptimalTrajectory

	initialStateTarget_S = adimensionalSolution['y'][0, 0:6]
	initialStateChaser_S = adimensionalSolution['y'][0, 6:12]
	DeltaIC_S = initialStateChaser_S - initialStateTarget_S

	# PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
	sol = adimensionalSolution['y']

	relDynami_LVLH = np.zeros((len(adimensionalSolution['x']), 6))

	for id in range(len(adimensionalSolution['x'])):
		rotatedRelativeState = ReferenceFrames.convert_S_to_LVLH(sol[id, 0:6], sol[id, 6:12] - sol[id, 0:6], param)
		relDynami_LVLH[id, 0:6] = rotatedRelativeState

	controlAction = adimensionalSolution['controlAction']
	
	# dimensionalize
	relDynami_LVLH *= param.xc
	relDynami_LVLH[:,3:] /= param.tc

	# FIGURE 1
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# if the optimal trajectory is defined
	if OBoptimalTrajectory is not None and "x" in OBoptimalTrajectory:
		OBOT = OBoptimalTrajectory['x'] * param.xc
		ax.plot(OBOT[:, 0], OBOT[:, 1], OBOT[:, 2], 'm--', linewidth=1.2, label="Optimal Trajectory")

	ax.plot(relDynami_LVLH[:, 0], relDynami_LVLH[:, 1], relDynami_LVLH[:, 2], 'b', linewidth=1.2, label="Actual Trajectory")
	ax.plot(0, 0, 0, 'r*', linewidth=1)

	# ax.quiver(relDynami_LVLH[:, 0], relDynami_LVLH[:, 1], relDynami_LVLH[:, 2], controlAction[:, 0]/100, controlAction[:, 1]/100, controlAction[:, 2]/100, color='lightgreen', linewidth=0.8, arrow_length_ratio=0.08, label="Control Action")
	# downsampled quiver
	ce = int(len(relDynami_LVLH[:, 0])/500)
	ax.quiver(relDynami_LVLH[::ce, 0], relDynami_LVLH[::ce, 1], relDynami_LVLH[::ce, 2],
		   controlAction[::ce, 0]/10, controlAction[::ce, 1]/10, controlAction[::ce, 2]/10, color='#02e80a', linewidth=0.8, arrow_length_ratio=0.1, label="Control Action")

	ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)
	ax.quiver(0, 0, 0, 0, 1, 0, color='r', linewidth=1)
	ax.quiver(0, 0, 0, 0, 0, 1, color='r', linewidth=1)

	# initial condition
	ax.plot(relDynami_LVLH[0, 0], relDynami_LVLH[0, 1], relDynami_LVLH[0, 2], color='k', linewidth=1) # arrow pointing to initial state
	ax.plot(relDynami_LVLH[0, 0], relDynami_LVLH[0, 1], relDynami_LVLH[0, 2], 'ok', ms = 3.5, label="Initial Position")

	# final condition
	holdState = param.holdingState * param.xc
	ax.plot(holdState[0], holdState[1], holdState[2], 'db', ms = 3.5, label="Hold Position")

	if phaseID == 1:
		plotConstraintsVisualization(1e3, 'SPHERE', 'yellow')
		plotConstraintsVisualization(200, 'SPHERE')
		plotConstraintsVisualization(2.5e3, 'SPHERE', '#808080')
	elif phaseID == 2:
		plotConstraintsVisualization(1e3, 'CONE')

	ax = plt.gca()
	ax.legend(loc='best')
	ax.set_title("Relative Dynamics [CONTROLLED]")
	ax.grid(True)
	ax.set_box_aspect([1,1,1])  # Equal aspect ratio
	plt.xlabel("R-BAR [km]")
	plt.ylabel("V-BAR [km]")
	plt.gca().set_zlabel("H-BAR [km]")
	plt.axis('equal')


	################################################
	## Plot control actions and relative dynamics ##
	################################################
	t = adimensionalSolution['x'] * param.tc / 60
	fig, axs = plt.subplots(3, 1) # , figsize=(8, 12)

	axs[0].plot(t, controlAction, linewidth=1.1)
	axs[0].grid(True)
	axs[0].set_title("Control Action [LVLH]")
	axs[0].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[0].set_xlabel("Time [min]")
	axs[0].set_ylabel("Control Action [-]")

	axs[1].plot(t, relDynami_LVLH[:, 0:3], linewidth=1)
	axs[1].set_title("Controlled Relative Dynamics [LVLH]")
	axs[1].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[1].grid(True)
	axs[1].set_xlabel("Time [min]")
	axs[1].set_ylabel("Position [km]")

	axs[2].plot(t, relDynami_LVLH[:, 3:6]*1e3, linewidth=1)
	axs[2].set_title("Controlled Relative Velocity [LVLH]")
	axs[2].legend(["R-BAR", "V-BAR", "H-BAR"], loc='best')
	axs[2].grid(True)
	axs[2].set_xlabel("Time [min]")
	axs[2].set_ylabel("Velocity [m/s]")

	plt.tight_layout()
	plt.show()


#########################################################################

def plotConstraintsVisualization( DeltaIC_meters, type='CONE', colore=None):
	# coefficients definition
	acone = 0.08
	bcone = 5

	if DeltaIC_meters < 200:
		rsphere = 200  # [m]
	else:
		rsphere = DeltaIC_meters

	if type.upper() == 'SPHERE':
		plot_sphere(rsphere, colore)
	elif type.upper() == 'CONE':
		plot_cone(DeltaIC_meters, acone, bcone, colore)


def plot_sphere(rsphere, colore='red'):
	rsphere*=1e-3
	u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
	x = rsphere*np.cos(u)*np.sin(v)
	y = rsphere*np.sin(u)*np.sin(v)
	z = rsphere*np.cos(v)

	fig = plt.gcf()
	ax = fig.axes[0]
	ax.plot_surface(x,y,z, color=colore, alpha=0.4, edgecolor='none')


def plot_cone(DeltaIC_meters, acone, bcone, colore='none'):
	z = lambda RbarX, VbarX: (acone**2 * bcone**3 - 3 * acone**2 * bcone**2 * VbarX + 3 * acone**2 * bcone * VbarX**2 - acone**2 * VbarX**3 - RbarX**2)
	pointsR = np.linspace(-3 * DeltaIC_meters, 3 * DeltaIC_meters, 1001)
	pointsV = np.linspace(-DeltaIC_meters, 0, 1001)
	X, Y = np.meshgrid(pointsR, pointsV)

	halfCone = z(X, Y)  # [m]
	toll = max(np.max(np.diff(pointsR)), np.max(np.diff(pointsV)))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if halfCone[i,j] < 0:
				halfCone[i,j] = np.nan
			else:
				if -(Y[i, j] - toll - bcone)**3 > (X[i, j]**2 / acone**2) and -(Y[i, j] + toll - bcone)**3 < (X[i, j]**2 / acone**2):
					halfCone[i, j] = 0
				else:
					halfCone[i,j] = np.sqrt(halfCone[i,j])

	fig = plt.gcf()
	ax = fig.axes[0]
	ax.plot_surface(X * 1e-3, Y * 1e-3, halfCone * 1e-3, color=colore, alpha=0.5, edgecolor='none')
	ax.plot_surface(X * 1e-3, Y * 1e-3, -halfCone * 1e-3, color=colore, alpha=0.5, edgecolor='none')