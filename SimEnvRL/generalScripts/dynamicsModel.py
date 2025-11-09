import numpy as np

## CR3BP ## 
def CR3BP(t, state, param, controlAction=None, disturbAction=None):
	"""
	dstate = CR3BP(t, state, param, controlAction, disturbAction)

	INPUTS:
		state 	  : is the [x y z vx vy vz] of the satellite
		param	  : is a dataclass that includes several simulation parameters among which
						massRatio = m2 / (m1 + m2)
		controlAction : contains the adimensionalized control action to be added in the integration
		disturbAction : contains the adimensionalized environmental disturbances to be added in the integration
	
	OUTPUTS:
		dstate	  : is the derivative of the state
		
	"""
	
	# If a control action is not present, set it to zero
	if controlAction is None:
		controlAction = np.array([0, 0, 0])
	if disturbAction is None:
		disturbAction = np.array([0, 0, 0])

	x = state[0]
	y = state[1]
	z = state[2]
	massRatio = param.massRatio

	r1 = np.sqrt((x + massRatio)**2 + y**2 + z**2)
	r2 = np.sqrt((x - 1 + massRatio)**2 + y**2 + z**2)

	# EQUATIONS OF MOTION
	vx = state[3]
	vy = state[4]
	vz = state[5]
	ax = (2 * vy + x - 
		  (1 - massRatio) / r1**3 * (x + massRatio) - 
		  massRatio / r2**3 * (x - 1 + massRatio) + 
		  controlAction[0] + disturbAction[0])
	ay = (-2 * vx + y - 
		  y * ((1 - massRatio) / r1**3 + massRatio / r2**3) + 
		  controlAction[1] + disturbAction[1])
	az = (-z * ((1 - massRatio) / r1**3 + massRatio / r2**3) + 
		  controlAction[2] + disturbAction[2])

	# DIFFERENTIAL OF STATE
	dstate = np.array([vx, vy, vz, ax, ay, az])

	return dstate

## Moon Frame CR3BP ##
def CR3BP_MoonFrame(t, state_M, param):
	"""
	This function computes the derivative of the state for the CR3BP,
	with respect to a reference frame which is Synodic but centered about
	the moon. For reference, see Franzini PhD thesis [T14].

	"""

	# extraction of data
	massRatio = param.massRatio
	rem = np.array([-1, 0, 0])
	rei = state_M[0:3] + rem  # distance from Earth in Moon Frame
	rmi = state_M[0:3]
	vmi = state_M[3:6]

	omegaMI = np.array([0, 0, 1]) # definition of Moon Synodic angular velocity wrt inertial RF

	# computation of the acceleration
	amiIner = -massRatio * rmi / np.linalg.norm(rmi)**3 - (1 - massRatio) * (rei / np.linalg.norm(rei)**3 - rem)
	ami = amiIner - 2 * np.cross(omegaMI, vmi) - np.cross(omegaMI, np.cross(omegaMI, rmi))

	# relative state differential definition
	dRelState = np.hstack((vmi, ami))
	
	return dRelState

## COMPUTE ENVIRONMENT DISTURBANCES ##
def computeEnvironmentDisturbances(t,SCparam,param):
	"""
	This function computes the environmental disturbances acting on a satellite

	INPUTS:
		t: time
		SCparam: dictionary containing the satellite parameters
		param: dataclass containing the simulation parameters

	OUTPUTS:
		distAcceleration_S: adimensionalized acceleration due to environmental disturbances

	"""
	
	# If the sunInitialAngle is not present, set it to zero radians
	if 'sunInitialAngle' not in param.__dict__:
		sunInitialAngle = 0
	else:
		sunInitialAngle = param.sunInitialAngle

	# set Theta as constant (for now)
	# NOTE: this is a simplification, since the simulation time is very short wrt the orbital period
	Theta = sunInitialAngle # to add the time varying component, add +1.996437750711854e-07*t
	sunVersor = np.array([np.cos(Theta), np.sin(Theta), 0])

	# compute the acceleration due to solar radiation pressure
	distAcceleration_S = (param.SolarFlux * SCparam["Area"] / SCparam["mass"] * 
						(1 + SCparam["reflCoeffSpecular"] + 2/3 * SCparam["reflCoeffDiffuse"])) * sunVersor


	# add environment noise
	

	
	# adimensionalize:
	distAcceleration_S = distAcceleration_S * (1e-3) * param.tc**2 / param.xc
	
	return distAcceleration_S