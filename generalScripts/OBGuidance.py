from scipy.integrate import solve_ivp
import time
import numpy as np
from generalScripts.CR3BP_MoonFrame import CR3BP_MoonFrame
from generalScripts.ReferenceFrames import versorsLVLH

def OBGuidance(relativeState_L, targetState_M, param):
	# This function implements the guidance algorithm for the optimal control of the chaser
	# spacecraft in the presence of obstacles. The algorithm is composed of two nested loops
	# the first loop implements the ASRE algorithm, while the second loop implements the APF
	# algorithm
	# 
	# INPUTS:  
	#   - relativeState_L: starting relative state in the LVLH frame
	#   - targetState_M: starting target state in the Moon centered frame
	#   - param: physical parameters of the system (given as dataclass)
	#
	# OUTPUTS:
	#   - controlAction: the control action to be applied to the chaser in LVLH frame

	# First loop: ASRE algorithm
	if trigger:  # Recompute loop 1
		controlAction = loopOne(relativeState_L, targetState_M, param)

	# Second loop: APF algorithm
	controlAction = loopTwo()

	return controlAction





def loopOne(relativeState_L, targetState_M, param):
	# This function implements the ASRE algorithm for the computation of an optimal trajectory
	# that allows to minimize the control effort while reaching the target
	# NOTE: this function does NOT consider any obstacle avoidance strategy / constraint
	# 
	# INPUTS:  
	#   - relativeState_L: starting relative state in the LVLH frame
	#   - targetState_M: starting target state in the Moon centered frame
	#   - param: physical parameters of the system (given as dataclass)
	#
	# OUTPUTS:
	#   - controlAction: the control action to be applied to the chaser in LVLH frame

	
	print("Computing Optimal Trajectory... ", end="")
	import time
	start_time = time.time()
	
	OBoptimalTrajectory = loopOne(relativeState_L, targetState_M, param)
	OBoptimalTrajectory['ref'] = OBoptimalTrajectory['x']
	
	elapsed_time = time.time() - start_time
	print(f"done. [Elapsed Time: {elapsed_time:.3f} sec]")

	return OBoptimalTrajectory



def loopTwo():
	# This function implements the second loop for the guidance algorithm, in particular
	# this loop allows to consider the presence of obstacles in the environment and possible
	# constraints. This loop implements the APF algorithm


	return controlAction





""" ASRE algorithm """
def ASRE(TOF, initialRelativeState_L, initialStateTarget_M, param):

	execution_time_start = time.time()

	# PARAMETERS
	t_i = 0				   # Initial time
	t_f = TOF			   # Final time
	N = 200				   # Number of time steps

	# TIME GRID
	tvec = np.linspace(t_i, t_f, N)

	# INITIAL AND FINAL STATES
	x_i = initialRelativeState_L
	x_f = np.array([0, 0, 0, 0, 0.6e-3 * param.tc / param.xc, 0])  # Final relative state
	u_guess = np.zeros((3, N-1))  # Initial control guess (zeros)

	# COST MATRICES
	Q = np.eye(6)
	R = 10 * np.eye(3)

	# INITIALIZE ITERATION
	x_guess = interpolateTrajectory(x_i, x_f, tvec)
	converged = False

	# ITERATIVE ASRE PROCESS
	A = computeA(initialStateTarget_M, param)
	B = computeB()

	phi_xx = np.eye(6)
	phi_yy = np.eye(6)
	phi_xy = np.zeros((6, 6))
	phi_yx = np.zeros((6, 6))
	PHI0 = np.block([[phi_xx, phi_xy], [phi_yx, phi_yy]])

	# Solve ODE for PHI
	PHIT_initial = np.concatenate([PHI0.ravel(), initialStateTarget_M])
	sol = solve_ivp(lambda t, PHIT: computePHIT(t, PHIT, B, Q, R, param), [tvec[0], tvec[-1]], PHIT_initial, t_eval=tvec)

	PHIT = sol.y.T
	PHI = PHIT[-1, :144].reshape(12, 12)
	lambda_i = np.linalg.solve(PHI[:6, 6:], x_f - PHI[:6, :6] @ x_i)

	x_new = np.zeros((6, N))
	u_new = np.zeros((3, N))
	x_new[:, 0] = x_i
	u_new[:, 0] = -np.linalg.solve(R, B.T @ lambda_i)

	for time_id in range(1, N):
		PHI = PHIT[time_id, :144].reshape(12, 12)
		lambda_ = PHI[6:, :6] @ x_i + PHI[6:, 6:] @ lambda_i
		x_new[:, time_id] = PHI[:6, :6] @ x_i + PHI[:6, 6:] @ lambda_i
		u_new[:, time_id] = -np.linalg.solve(R, B.T @ lambda_)

	optimalTrajectory = {"t": tvec, "x": x_new, "u": u_new}

	execution_time = time.time() - execution_time_start
	print(f"ASRE Finished the optimization. [Execution time: {execution_time:.2f} sec]")
	print(f" > Estimated error in Target capture: {np.linalg.norm(optimalTrajectory['x'][:3, -1] - x_f[:3]) * 1e3 * param.xc:.3f} meters.")

	return optimalTrajectory


def interpolateTrajectory(x_i, x_f, tvec):
	x_guess = np.zeros((len(x_i), len(tvec)))
	for i in range(len(x_i)):
		testv = np.linspace(x_i[i], x_f[i], len(tvec))
		x_guess[i, :] = testv
	return x_guess

def computeA(targetState_M, param):
	return relDynOBmatrixA(1, targetState_M, param)


def computeB():
	B = np.zeros((6, 3))
	B[3:6, :] = np.eye(3)
	return B


def computePHIT(t, PHIT, B, Q, R, param):
	PHI = PHIT[:144].reshape(12, 12)
	targetState_M = PHIT[144:]
	dST = CR3BP_MoonFrame(t, targetState_M, param)
	A = relDynOBmatrixA(t, targetState_M, param)
	M = np.block([[A, -B @ np.linalg.solve(R, B.T)], [-Q, -A]])
	DP = (M @ PHI).ravel()
	return np.concatenate([DP, dST])


def relDynOBmatrixA(t, targetState_M, param):
	massRatio = param.massRatio
	omegaMI = np.array([0, 0, 1])
	rem = np.array([-1, 0, 0])

	rTM = targetState_M[:3]
	rTE = rTM + rem
	vTM = targetState_M[3:]

	rTMn = np.linalg.norm(rTM)
	vTMn = np.dot(rTM, vTM) / rTMn
	rTEn = np.linalg.norm(rTE)

	eR_x, eV_y, eH_z, _, _, _ = versorsLVLH(targetState_M, param)
	RotMat_M_to_L = np.vstack([eR_x, eV_y, eH_z])

	dSt = CR3BP_MoonFrame(t, targetState_M, param)
	aTM = dSt[3:]

	hTM = np.cross(rTM, vTM)
	hTM_norm = np.linalg.norm(hTM)

	omegaLM = np.array([rTMn / hTM_norm * np.dot(aTM, eH_z), 0, 1 / rTMn * np.dot(vTM, eV_y)])
	Xi = -massRatio / rTMn ** 3 * (np.eye(3) - 3 * np.outer(rTM, rTM) / rTMn ** 2)
	Amat = np.block([[np.zeros((3, 3)), np.eye(3)], [Xi, np.zeros((3, 3))]])
	return Amat
