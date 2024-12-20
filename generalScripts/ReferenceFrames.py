import numpy as np
from .CR3BP_MoonFrame import *

def versorsLVLH(targetState_M, param):
	"""
	versorsLVLH(targetState_M, param)
	
	This function computes the LVLH versors in the Moon-centered frame
	and its time derivatives. This function can be used to build the
	rotation matrices to rotate from Moon-centered Synodic to LVLH and
	vice versa.
	"""

	# Extract Moon-centered synodic values
	rTM = targetState_M[0:3]
	vTM = targetState_M[3:6]
	dSt = CR3BP_MoonFrame(0, targetState_M[0:6], param)
	aTM = dSt[3:6]

	# Compute other values
	hTM = np.cross(rTM, vTM, axis=0)
	hTM_norm = np.linalg.norm(hTM, axis=0)

	# Compute new reference frame axes (LVLH)
	eR_x = rTM / np.linalg.norm(rTM, axis=0)
	eH_z = np.cross(rTM, vTM, axis=0) / hTM_norm
	eV_y = np.cross(eH_z, eR_x, axis=0)

	# Derivatives of the reference frame axes (LVLH)
	eR_x_dot = (1 / np.linalg.norm(rTM, axis=0)) * (np.dot(vTM.T, eV_y) * eV_y)
	eH_z_dot = -(np.linalg.norm(rTM, axis=0) / hTM_norm) * (np.dot(aTM.T, eH_z) * eV_y)
	eV_y_dot = (np.linalg.norm(rTM, axis=0) / hTM_norm) * (np.dot(aTM.T, eH_z) * eH_z) - \
			   (1 / np.linalg.norm(rTM, axis=0)) * (np.dot(vTM.T, eV_y) * eR_x)

	return eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot


def computeRotationMatrixLVLH(target_state_M, param):
    """
	This funciton computes the rotation matrix from moon centered synodic
	to the LVLH RF centered about the target
	"""
    # computing new reference frame axis (LVLH)
    eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot = versorsLVLH(target_state_M, param)

    # Rotational matrices
    R = np.vstack([eR_x.flatten(), eV_y.flatten(), eH_z.flatten()])
    Rdot = np.vstack([eR_x_dot.flatten(), eV_y_dot.flatten(), eH_z_dot.flatten()])
    
	# return the rotation matrix and its derivative
    return R, Rdot


def rotate_S_to_LVLH(targetState_S, stateToBeRotated_S, param):
    """
    rotate_S_to_LVLH rotates the stateToBeRotated_S from S to LVLH
    To rotate from S to LVLH first a translation is needed,
    then it is possible to rotate from S to M and eventually rotate from M to LVLH
	"""

    # Translate from Synodic to Moon
    rM = np.array([1 - param.massRatio, 0, 0])  # Position of the moon in Synodic frame
    target_state_M = targetState_S - np.concatenate([rM, np.zeros(3)])

    # Rotating from Moon to Moon Synodic [T14]
    FranziRot = np.array([[-1, 0, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0],
                          [0, 0, +1, 0, 0, 0],
                          [0, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, -1, 0],
                          [0, 0, 0, 0, 0, +1]])
    target_state_M = FranziRot @ target_state_M
    state_to_be_rotated_M = FranziRot @ stateToBeRotated_S

    # Rotating frame from M to LVLH
    rotated_state, _ = rotate_M_to_LVLH(target_state_M, state_to_be_rotated_M, param)
    
    return rotated_state


def rotate_M_to_LVLH(target_state_M, state_to_be_rotated, param):
    """
    this function rotates a state from M to LVLH
     
	"""
    # Compute LVLH versors from Moon centered synodic
    eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot = versorsLVLH(target_state_M, param)

    # Rotating matrices
    R = np.vstack([eR_x, eV_y, eH_z])  # 3x3 rotation matrix
    Rdot = np.vstack([eR_x_dot, eV_y_dot, eH_z_dot]) # 3x3 derivative of rotation matrix
    RTOT = np.block([[R, np.zeros((3, 3))], [Rdot, R]])

    # Rotating frame
    rotated_state = (RTOT @ state_to_be_rotated).transpose()
    
    return rotated_state, RTOT