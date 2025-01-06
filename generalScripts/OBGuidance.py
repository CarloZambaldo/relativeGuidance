import numpy as np
from generalScripts import dynamicsModel, ReferenceFrames
from scipy.integrate import solve_ivp
import time

## OBGuidance ##################################################################################
def OBGuidance(envTime,OBrelativeState,OBtargetState,phaseID,param,trigger=None,OBoptimalTrajectory=None):

	# extract parameters
	Umax = param.maxAdimThrust

	# Define phase-specific constraints and target state
	match phaseID:
		case 1:
			constraintType = 'SPHERE'
			aimAtState = param.holdingState
			characteristicSize = 200
		case 2:
			constraintType = 'CONE'
			aimAtState = param.dockingState
			characteristicSize = {'acone': 0.08, 'bcone': 5}
		case _:
			raise ValueError("Wrong phaseID")

	# Loop 1: Optimal Trajectory Computation
	if trigger: # if triggered, compute optimal trajectory
		OBoptimalTrajectory = loopOne(envTime, OBrelativeState, OBtargetState, aimAtState, phaseID, param)
		trigger = 0
		if OBoptimalTrajectory: # if trajectory is not empty, check for constraint violation
			constraintViolationFlag = checkConstraintViolation(OBoptimalTrajectory, constraintType, characteristicSize)
			if constraintViolationFlag:
				print("Warning: Constraints could be violated with the given trajectory.\n")

	# Loop 2: Surface Computation
	closestOptimalControl, surface_L1_pos, surface_L1_vel, surface_L2 = loopTwo(
		envTime, OBrelativeState, aimAtState, OBoptimalTrajectory, constraintType, param)

	# Compute sliding surface
	sigma = surface_L2 + 1e-3 * (1e3 * surface_L1_vel + surface_L1_pos)

	# Compute control action (using ASRE+APF+SMC)
	controlAction_L = closestOptimalControl - Umax * np.tanh(sigma)
	return controlAction_L, OBoptimalTrajectory


# Loop 1: Optimal Trajectory
def loopOne(envTime, initialRelativeState_L, initialTargetState_M, aimAtState, phaseID, param):
	print("Computing Optimal Trajectory... ", end='')
	
	TOF = np.linalg.norm(initialRelativeState_L[:3]) / (1e-3 / param.xc * param.tc) * 1.1
	
	### TODO: HERE CAN ADD A try catch block to handle the case where the optimal trajectory is not feasible
	
	if TOF > 0:
		print(f"\n  Estimated TOF: {TOF} [-]")
		exectime_start = time.time()
		optimalTrajectory = ASRE(envTime, TOF, initialRelativeState_L, initialTargetState_M, aimAtState, phaseID, param)
		print(f" done. [Elapsed Time: {time.time() - exectime_start} sec]")
	else:
		print("\n  Estimated TOF is too small. OBoptimalTrajectory is set to empty.")
		optimalTrajectory = None
	
	return optimalTrajectory


# Loop 2: Surface Computation
def loopTwo(envTime, relativeState, aimAtState, OBoptimalTrajectory, constraintType, param):

	if OBoptimalTrajectory: # if the optimal trajectory exists, use it to compute the closest optimal state
		interpTime = envTime - OBoptimalTrajectory['envStartTime']
		if interpTime < 0:
			raise ValueError("Error in interpTime definition. Possibly due to numerical integrator.")
	else:
		interpTime = None
	
	if OBoptimalTrajectory and 'state' in OBoptimalTrajectory and interpTime <= OBoptimalTrajectory['time'][-1]:
		closestOptimalState = np.array([
			np.interp(interpTime, OBoptimalTrajectory['time'], OBoptimalTrajectory['state'][:,i])
			for i in range(6)
		])
		closestOptimalControl = np.array([
			np.interp(interpTime, OBoptimalTrajectory['time'], OBoptimalTrajectory['controlAction'][:,i])
			for i in range(3)
		])
		print(f"  [envTime {envTime}] closestOptimalState [dist = {np.linalg.norm(relativeState[:3] - closestOptimalState[:3]) * 1e3 * param.xc} m; |deltaV| = {np.linalg.norm(closestOptimalState[3:6] - relativeState[3:6]) * 1e3 * param.xc / param.tc} m/s]")
	else:
		closestOptimalState = aimAtState
		closestOptimalControl = np.zeros(3)
		print(f"  [envTime {envTime}] <info> Using aimAtState as convergence point [dist = {np.linalg.norm(relativeState[:3] - closestOptimalState[:3]) * 1e3 * param.xc} m; |deltaV| = {np.linalg.norm(relativeState[3:6] - closestOptimalState[3:6]) * 1e3 * param.xc / param.tc} m/s]")

	surface_L1_pos = (relativeState[:3]  - closestOptimalState[:3])  * 1e3 * param.xc
	surface_L1_vel = (relativeState[3:6] - closestOptimalState[3:6]) * 1e3 * param.xc / param.tc

	_, surface_L2 = APF(relativeState, constraintType, param)
	return closestOptimalControl, surface_L1_pos, surface_L1_vel, surface_L2



## ASRE ##################################################################################
def ASRE(timeNow, TOF, initialRelativeState_L, initialStateTarget_M, finalAimState, phaseID, param):
	exectime_start = time.time()
	
	# PARAMETERS
	t_i = 0                  # Initial time
	t_f = TOF                # Final time
	N = 200                  # Number of time steps

	# TIME GRID DEFINITON
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
	#A = computeA(1, initialStateTarget_M, param)
	B = computeB()

	phi_xx = np.eye(6)
	phi_yy = np.eye(6)
	phi_xy = np.zeros((6, 6))
	phi_yx = np.zeros((6, 6))

	PHI0 = np.block([[phi_xx, phi_xy], [phi_yx, phi_yy]])
	initial_conditions = np.concatenate([PHI0.flatten(), initialStateTarget_M])

	#PHIT = odeint(compute_PHIT, initial_conditions, tvec, args=(B, Q, R, param))
	solution = solve_ivp(compute_PHIT, [t_i, t_f], initial_conditions, args=(B,Q,R,param), t_eval=tvec, method='RK45')
	PHIT = solution.y.T
	PHI = PHIT[-1, :144].reshape(12, 12)

	lambdaCoeff_i = np.linalg.solve(PHI[:6, 6:], (x_f - PHI[:6, :6] @ x_i))

	x_new = np.zeros((6, N))
	u_new = np.zeros((3, N))
	x_new[:, 0] = x_i
	u_new[:, 0] = -np.linalg.solve(R, B.T @ lambdaCoeff_i)

	# for each time step (compute the trajectory)
	for time_id in range(1, N):
		PHI = PHIT[time_id, :144].reshape(12, 12)
		lambdaCoeff = PHI[6:, :6] @ x_i + PHI[6:, 6:] @ lambdaCoeff_i
		x_new[:, time_id] = PHI[:6, :6] @ x_i + PHI[:6, 6:] @ lambdaCoeff_i
		u_new[:, time_id] = -np.linalg.solve(R, B.T @ lambdaCoeff)

	# Update the guess for the next iteration
	x_guess = x_new
	u_guess = u_new

	# OUTPUT OPTIMAL TRAJECTORY
	optimalTrajectory = {
		"time": tvec,
		"state": x_guess,
		"controlAction": u_guess,
		"envStartTime": timeNow
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

def computeA(t, targetState_M, param):
	# EXTRACTING VALUES FROM INPUT
	# environment data [in Moon Synodic]
	massRatio = param.massRatio
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
	eR_x, eV_y, eH_z, _, _, _ = ReferenceFrames.versorsLVLH(targetState_M, param)
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
	
	derivataStrana = lambda q: 1 / np.linalg.norm(q)**3 * (np.eye(3) - 3 * np.outer(q, q) / np.linalg.norm(q)**2)
	# def derivataStrana(q):
	# 	return 1 / np.linalg.norm(q)**3 * (np.eye(3) - 3 * np.outer(q, q) / np.linalg.norm(q)**2)

	return Amat

def computeB():
	# Computes control influence matrix B(state).
	B = np.zeros((6, 3))
	B[3:, :] = np.eye(3)
	return B

def compute_PHIT(t, PHIT, B, Q, R, param):
	PHI = PHIT[:144].reshape(12, 12)
	targetState_M = PHIT[144:150]

	# Target state and system dynamics retrieval
	dST = dynamicsModel.CR3BP_MoonFrame(t, targetState_M, param)
	A = computeA(t, targetState_M, param)

	# PHI computation
	M = np.block([[A, -B @ np.linalg.solve(R, B.T)], [-Q, -A]])
	DP = M @ PHI
	DP = DP.flatten()

	return np.concatenate([DP, dST]).flatten() # TODO check if .flatten() is correct

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
			h = lambda r: r[0]**2 + acone**2 * (r[1] - bcone)**3 + r[2]**2

			# computation of the nablas
			Nablah = lambda r: np.array([2 * r[0], 3 * acone**2 * (r[1] - bcone)**2, 2 * r[2]])

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

## CHECK CONTRAINTS VIOLATION ##################################################################################
def checkConstraintViolation(OBoptimalTrajectory, constraintType, characteristicSize):

	"""
	function [violationFlag,violationPosition] = checkConstraintViolation(trajectory,contraintType,characteristicSize)
		% 
		violationFlag = false;
		
		if isfield(trajectory,"x")
			if isfield(trajectory,"u")
				controlAction = trajectory.u;
				violationPosition = zeros(2,length(controlAction));
				for idx = 1:length(controlAction)
					if norm(controlAction(:,idx)) > 1e-2
						violationFlag = true;
						violationPosition(2,idx) = 1;
						%warning("Violation of Thrust Constraint");
					end
				end
			end
			trajectory = trajectory.x;
		else
			trajectory = trajectory.x;
			violationPosition = zeros(1,size(trajectory,2));
		end
		if size(trajectory,1)>6
			trajectory = trajectory';
		end
		switch upper(contraintType)
			case 'SPHERE'
				for idx=1:size(trajectory,2)
					if trajectory(1,idx)^2+trajectory(2,idx)^2+trajectory(3,idx)^2 <= characteristicSize^2
						violationFlag = true;
						violationPosition(1,idx) = 1;
					end
				end
			case 'CONE'
				for idx=1:size(trajectory,2)
					if  characteristicSize.acone^2*(trajectory(2,idx)^2-characteristicSize.bcone)^3 + trajectory(1,idx)^2 + trajectory(3,idx)^2 > 0
						violationFlag = true;
						violationPosition(1,idx) = 1;
					end
				end
		end
		
		if violationFlag
			warning("The computed Trajectory violates the constraints.");
		else
			fprintf("No violations of the constraints identified.\n");
		end

	end
	"""