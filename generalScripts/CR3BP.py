import numpy as np

def CR3BP(t, state, param, controlAction=None, disturbAction=None):
	"""
	dstate = CR3BP(t, state, param, controlAction)

	state 	  : is the [x y z vx vy vz] of the satellite
	param	  : is a dataclass that includes several simulation parameters among which
					massRatio = m2 / (m1 + m2)
	controlAction : contains the adimensionalized control action to be added in the integration
	disturbAction : contains the adimensionalized environmental disturbances to be added in the integration

	-----------------------
	Last Update: 18/12/2024
	-----------------------
	
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

