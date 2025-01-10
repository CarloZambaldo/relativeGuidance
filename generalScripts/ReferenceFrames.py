import numpy as np
from .dynamicsModel import *

def versorsLVLH(targetState_M, param):
	"""
	versorsLVLH(targetState_M, param)
	
	This function computes the LVLH versors in the Moon-centered frame
	and its time derivatives. This function can be used to build the
	rotation matrices to rotate from Moon-centered Synodic to LVLH and
	vice versa.
     
     NOTE: output versors are ROW vectors
	"""

	# Extract Moon-centered synodic values
	rTM = targetState_M[0:3]
	vTM = targetState_M[3:6]
	dSt = CR3BP_MoonFrame(0, targetState_M[0:6], param)
	aTM = dSt[3:6]

	# Compute other values
	hTM = np.cross(rTM, vTM)
	hTM_norm = np.linalg.norm(hTM)

	# Compute new reference frame axes (LVLH)
	eR_x = rTM / np.linalg.norm(rTM)
	eH_z = np.cross(rTM, vTM) / hTM_norm
	eV_y = np.cross(eH_z, eR_x)

	# Derivatives of the reference frame axes (LVLH)
	eR_x_dot = (1 / np.linalg.norm(rTM)) * (np.dot(vTM, eV_y) * eV_y)
	eH_z_dot = -(np.linalg.norm(rTM) / hTM_norm) * (np.dot(aTM, eH_z) * eV_y)
	eV_y_dot = np.cross(eH_z_dot,eR_x) + np.cross(eH_z,eR_x_dot) #(np.linalg.norm(rTM) / hTM_norm) * (np.dot(aTM.T, eH_z) * eH_z) - (1 / np.linalg.norm(rTM)) * (np.dot(vTM.T, eV_y) * eR_x)

	return eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot


## 
def computeRotationMatrixLVLH(targetState_M, param):
    """
	This funciton computes the rotation matrix
    FROM moon centered synodic
	TO the LVLH RF centered about the target
	"""
    # computing new reference frame axis (LVLH)
    eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot = versorsLVLH(targetState_M, param)

    # Rotational matrices
    R = np.vstack([eR_x.flatten(), eV_y.flatten(), eH_z.flatten()])
    Rdot = np.vstack([eR_x_dot.flatten(), eV_y_dot.flatten(), eH_z_dot.flatten()])
    
	# return the rotation matrix and its derivative
    return R, Rdot


##
def convert_S_to_LVLH(targetState_S, stateToBeRotated_SCM, param):
    """
    convert_S_to_LVLH rotates the stateToBeRotated_S from S to LVLH
    To rotate from S to LVLH first a translation is needed,
    then it is possible to rotate from S to M and eventually rotate from M to LVLH

    NOTE: the stateToBeRotated_SCM is ONLY ROTATED NOT TRASLATED from the COG of Earth-Moon
    to Moon center  

	"""

    # Translate from Synodic to Moon centered (still not Franzini RF)
    # Rotating from Moon to Moon Synodic [T14] (FranziRot)
    targetState_M = np.array([-targetState_S[0]+(1-param.massRatio),-targetState_S[1],targetState_S[2],
                              -targetState_S[3],-targetState_S[4],targetState_S[5]])
    stateToBeRotated_M = np.array([-stateToBeRotated_SCM[0],-stateToBeRotated_SCM[1],stateToBeRotated_SCM[2],
                                   -stateToBeRotated_SCM[3],-stateToBeRotated_SCM[4],stateToBeRotated_SCM[5]])

    # Rotating frame from M to LVLH
    rotated_state, _ = convert_M_to_LVLH(targetState_M, stateToBeRotated_M, param)
    
    return rotated_state


##
def convert_M_to_LVLH(target_state_M, stateToBeRotated_M, param):
    """
    this function rotates a state from M to LVLH

    NOTE: the stateToBeRotated_M shall already be in the moon centered synodic frame M

	"""

    R, Rdot = computeRotationMatrixLVLH(target_state_M, param)#okVV
    RTOT = np.block([[R, np.zeros((3, 3))], [Rdot, R]])#okVV

    # Rotating frame
    rotated_state = RTOT @ stateToBeRotated_M
    
    return rotated_state, RTOT


##ok
def convert_LVLH_to_S(targetState_S,stateToBeRotated_L,param):
    """
    convert_LVLH_to_S rotates the stateToBeRotated_L from LVLH to S
    To rotate from LVLH to S : first a rotation from LVLH to M is needed, then it is
    possible to rotate from M to S (FranziRot)

    NOTE: targetState_S is given in S, therefore traslation and rotation to M are required 

    NOTE: the stateToBeRotated_L is ONLY ROTATED NOT TRASLATED from MOON to the COG of Earth-Moon
    ( since the function is used for the relative states and not absolute states )
  
    """

    # translate from Synodic to Moon, Rotating from Moon to Moon Synodic [T14]
    targetState_M = np.array([-targetState_S[0]+(1-param.massRatio),-targetState_S[1],targetState_S[2],
                              -targetState_S[3],-targetState_S[4],targetState_S[5]])

    ## rotate from LVLH to M
    R, Rdot = computeRotationMatrixLVLH(targetState_M, param)
    RTOTMtoL = np.block([[R, np.zeros((3, 3))], [Rdot, R]]) # NOTE: this matrix must be inverted!!!

    # rotating frame
    rotatedState_M = np.linalg.solve(RTOTMtoL,np.eye(6)) @ stateToBeRotated_L

    # rotating from Moon to the Synodic (FranziRot)
    rotatedState_S = np.array([-rotatedState_M[0],-rotatedState_M[1],rotatedState_M[2],
                               -rotatedState_M[3],-rotatedState_M[4],rotatedState_M[5]])

    return rotatedState_S