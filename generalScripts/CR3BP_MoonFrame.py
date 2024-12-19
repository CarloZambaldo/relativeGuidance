import numpy as np

def CR3BP_MoonFrame(t, state_M, param):
	"""
	This function computes the derivative of the state for the CR3BP,
	with respect to a reference frame which is Synodic but centered about
	the moon. For reference, see Franzini PhD thesis [T14].

	-----------------------
	Last Update: 18/12/2024
	-----------------------
	"""

	# extraction of data
	massRatio = param.massRatio
	rem = np.array([-1, 0, 0])
	rei = state_M[0:3] + rem
	rmi = state_M[0:3]
	vmi = state_M[3:6]

	omegaMI = np.array([0, 0, 1]) # definition of Moon Synodic angular velocity wrt inertial RF

	# computation of the acceleration
	amiIner = -massRatio * rmi / np.linalg.norm(rmi)**3 - (1 - massRatio) * (rei / np.linalg.norm(rei)**3 - rem)
	ami = amiIner - 2 * np.cross(omegaMI, vmi) - np.cross(omegaMI, np.cross(omegaMI, rmi))

	# relative state differential definition
	dRelState = np.hstack((vmi, ami))
	
	return dRelState
