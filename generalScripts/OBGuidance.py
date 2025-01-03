import numpy as np
import CR3BP_MoonFrame
import ReferenceFrames

## ASRE ##################################################################################
import numpy as np
from scipy.integrate import odeint
import time

def ASRE(TOF, initialRelativeState_L, initialStateTarget_M, finalAimState, phaseID, param):
	exectime_start = time.time()
	
	# PARAMETERS
	t_i = 0                   # Initial time
	t_f = TOF                # Final time
	N = 200                  # Number of time steps

	# TIME GRID
	tvec = np.linspace(t_i, t_f, N)
	
	# INITIAL AND FINAL STATES
	x_i = initialRelativeState_L
	x_f = finalAimState.flatten()
	u_guess = np.zeros((3, N - 1))  # Initial control guess (zeros)

	# COST MATRICES
	match phaseID:
		case 1:
			Q = np.eye(6)
			R = np.eye(3)
		case 2:
			Q = np.block([
				[np.diag([1e12, 1e-5, 1e12]), np.zeros((3, 3))],
				[np.zeros((3, 3)), np.diag([1, 1e-6, 1])]
			])
			R = np.diag([0.9e-1, 1, 0.9e-1])
		case _:
			Q = np.eye(6)
			R = np.eye(3)

	# INITIALIZE ASRE
	x_guess = interpolate_trajectory(x_i, x_f, tvec)  # Linear initial guess

	# iteration 0
	A = compute_A(initialStateTarget_M, param)
	B = compute_B()

	phi_xx = np.eye(6)
	phi_yy = np.eye(6)
	phi_xy = np.zeros((6, 6))
	phi_yx = np.zeros((6, 6))

	PHI0 = np.block([[phi_xx, phi_xy], [phi_yx, phi_yy]])
	initial_conditions = np.concatenate([PHI0.flatten(), initialStateTarget_M])
	PHIT = odeint(compute_PHIT, initial_conditions, tvec, args=(B, Q, R, param))
	PHI = PHIT[-1, :144].reshape(12, 12)

	lambda_i = np.linalg.solve(PHI[:6, 6:], (x_f - PHI[:6, :6] @ x_i))

	x_new = np.zeros((6, N))
	u_new = np.zeros((3, N))
	x_new[:, 0] = x_i
	u_new[:, 0] = -np.linalg.solve(R, B.T @ lambda_i)

	# for each time step (compute the trajectory)
	for time_id in range(1, N):
		PHI = PHIT[time_id, :144].reshape(12, 12)
		lambda_ = PHI[6:, :6] @ x_i + PHI[6:, 6:] @ lambda_i
		x_new[:, time_id] = PHI[:6, :6] @ x_i + PHI[:6, 6:] @ lambda_i
		u_new[:, time_id] = -np.linalg.solve(R, B.T @ lambda_)

	# Update the guess for the next iteration
	x_guess = x_new
	u_guess = u_new

	# OUTPUT OPTIMAL TRAJECTORY
	optimalTrajectory = {
		"time": tvec,
		"state": x_guess,
		"controlAction": u_guess
	}

	# print the execution time of the simulation
	exectime = time.time() - exectime_start
	print(f"ASRE Converged. [Execution time: {exectime:.2f} sec]")
	return optimalTrajectory

def interpolate_trajectory(x_i, x_f, tvec):
	# Generates an initial guess for the state trajectory as a linear interpolation.
	x_guess = np.zeros((len(x_i), len(tvec)))
	for i in range(len(x_i)):
		x_guess[i, :] = np.linspace(x_i[i], x_f[i], len(tvec))
	return x_guess

def compute_A(targetState_M, param):
	# Computes state-dependent dynamics matrix A(state).
	return relDynOBmatrixA(1, targetState_M, param)

def compute_B():
	# Computes control influence matrix B(state).
	B = np.zeros((6, 3))
	B[3:, :] = np.eye(3)
	return B

def compute_PHIT(PHIT, t, B, Q, R, param):
	PHI = PHIT[:144].reshape(12, 12)
	targetState_M = PHIT[144:150]

	# Target state and system dynamics retrieval
	dST = CR3BP_MoonFrame(t, targetState_M, param)
	A = relDynOBmatrixA(t, targetState_M, param)

	# PHI computation
	M = np.block([[A, -B @ np.linalg.solve(R, B.T)], [-Q, -A]])
	DP = M @ PHI
	DP = DP.flatten()

	return np.concatenate([DP, dST])

def relDynOBmatrixA(t, targetState_M, param):
	# EXTRACTING VALUES FROM INPUT
	# environment data [in Moon Synodic]
	massRatio = param['massRatio']
	omegaMI = np.array([0, 0, 1])
	rem = np.array([-1, 0, 0])

	# target data
	rTM = targetState_M[:3]
	rTE = rTM + rem
	vTM = targetState_M[3:6]

	# compute norms
	rTMn = np.linalg.norm(rTM)
	vTMn = np.dot(rTM, vTM) / np.linalg.norm(rTM)
	rTEn = np.linalg.norm(rTE)

	# LVLH versors (RVH convention) and Rotation Matrix
	eR_x, eV_y, eH_z = ReferenceFrames.versorsLVLH(targetState_M, param)
	RotMat_M_to_L = np.array([eR_x, eV_y, eH_z])

	# computing aTM from the CR3BP from Franzini (Moon Centered)
	dSt = CR3BP_MoonFrame(t, targetState_M, param)
	aTM = dSt[3:6]

	# computation of angular momentum and derivatives
	hTM = np.cross(rTM, vTM)
	hTM_norm = np.linalg.norm(hTM)
	hTM_dot = np.cross(rTM, aTM)

	# ANGULAR VELOCITY COMPUTATION
	omegaLM = np.array([
		rTMn / hTM_norm * np.dot(aTM, eH_z),
		0,
		1 / rTMn * np.dot(vTM, eV_y)
	])

	# compute jerk
	JI = (-massRatio * derivataStrana(rTM) @ (vTM + np.cross(omegaMI, rTM))
		  - (1 - massRatio) * (derivataStrana(rTE) @ (vTM + np.cross(omegaMI, rTM) + np.cross(omegaMI, rem))
		  - derivataStrana(rem) @ np.cross(omegaMI, rem)))
	JTM = (JI - 3 * np.cross(omegaMI, aTM) 
		  - 3 * np.cross(omegaMI, np.cross(omegaMI, vTM)) 
		  - np.cross(omegaMI, np.cross(omegaMI, np.cross(omegaMI, rTM))))

	# ANGULAR ACCELERATION
	omegaLM_dot = np.array([
		rTMn / hTM_norm * (vTMn / rTMn * np.dot(aTM, eH_z)
		- 2 * rTMn / hTM_norm * np.dot(aTM, eV_y) * np.dot(aTM, eH_z)
		+ np.dot(JTM, eH_z)),
		0,
		1 / rTMn * (np.dot(aTM, eV_y) - 2 * vTMn / rTMn * np.dot(vTM, eV_y))
	])

	omegaLI = omegaLM + RotMat_M_to_L @ omegaMI
	omegaLI_dot = omegaLM_dot - np.cross(omegaLM, RotMat_M_to_L @ omegaMI)

	# RELATIVE ACCELERATION
	rTM = RotMat_M_to_L @ rTM
	rTE = RotMat_M_to_L @ rTE

	OMEGA_LI = np.array([
		[0, -omegaLI[2], omegaLI[1]],
		[omegaLI[2], 0, -omegaLI[0]],
		[-omegaLI[1], omegaLI[0], 0]
	])

	OMEGA_LI_dot = np.array([
		[0, -omegaLI_dot[2], omegaLI_dot[1]],
		[omegaLI_dot[2], 0, -omegaLI_dot[0]],
		[-omegaLI_dot[1], omegaLI_dot[0], 0]
	])

	Xi = (-massRatio / rTMn**3 * (np.eye(3) - 3 * np.outer(rTM, rTM) / rTMn**2)
		  - (1 - massRatio) / rTEn**3 * (np.eye(3) - 3 * np.outer(rTE, rTE) / rTEn**2))
	Amat = np.block([
		[np.zeros((3, 3)), np.eye(3)],
		[Xi - OMEGA_LI_dot - OMEGA_LI @ OMEGA_LI, -2 * OMEGA_LI]
	])

	return Amat

def derivataStrana(q):
	return 1 / np.linalg.norm(q)**3 * (np.eye(3) - 3 * np.outer(q, q) / np.linalg.norm(q)**2)


## APF ##################################################################################
def APF(relativeState_L, constraintType, param):
	# coefficients definition
	adi2meters = param.xc * 1e3
	Umax = param.maxAdimThrust

	# extraction of relative state and conversion to meters and meters per second
	rho = np.array(relativeState_L[:3]) * adi2meters
	v_rho = np.array(relativeState_L[3:6]) * adi2meters / param.tc

	match constraintType:
		case 'CONE':
			# constraints characteristic dimensions definition
			acone = 0.08  # note: these are adimensional parameters to have 0.9m of radius at docking port
			bcone = 5     # note: these are adimensional parameters to have 0.9m of radius at docking port

			# coefficients definition
			Krep = 1e-1

			# approach cone definition
			def h(r):
				return r[0]**2 + acone**2 * (r[1] - bcone)**3 + r[2]**2

			# computation of the nablas
			def Nablah(r):
				return np.array([2 * r[0], 3 * acone**2 * (r[1] - bcone)**2, 2 * r[2]])

			# potential fields computation (if contract is violated a constant repulsive field is applied),
			# otherwise the repulsive field is computed as the gradient of the repulsive potential
			if rho[0]**2 + rho[2]**2 >= -(acone**2 * (rho[1] - bcone)**3):  # if constraint is violated
				NablaU_APF = 1e1 * np.array([1, 0, 1]) * (rho / np.linalg.norm(rho))
			else:
				NablaU_APF = Krep * (rho / h(rho)**2 - (rho.T @ rho) * Nablah(rho) / h(rho)**3)

		case 'SPHERE':
			# constraints characteristic dimensions definition
			SphereRadius_SS = 2.5e3  # [m]

			# coefficients definition
			K_SS = np.array([1e-2, 1, 1e-2]) * 1e4

			# computation of the repulsive field (only when the constraint is violated)
			NablaUrep_SS = -rho / np.linalg.norm(rho) * (rho.T @ rho - SphereRadius_SS**2 <= 0)
			NablaU_APF = K_SS * NablaUrep_SS  # repulsive field in case of SPHERICAL constraint

		case _:
			raise ValueError("Constraint not defined properly.")

	# sliding surface definition and control action computation
	sigma = NablaU_APF
	controlAction = -Umax * np.tanh(sigma)

	return controlAction, sigma
