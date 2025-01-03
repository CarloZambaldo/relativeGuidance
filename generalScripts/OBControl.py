from generalScripts import ReferenceFrames
import numpy as np
def OBControl(targetState_M,uToBeRotated_L,param):
    """
    This function rotates the control action from LVLH to Synodic frame
    
    """

    # compute LVLH versors from Moon centered synodic
    eR_x, eV_y, eH_z, _ = ReferenceFrames.versorsLVLH(targetState_M,param)

    # Rotating matrices
    R_L_to_M = np.array([eR_x.flatten(), eV_y.flatten(), eH_z.flatten()]).T

    # Rotating from L to M
    rotatedControlAction = np.dot(R_L_to_M, uToBeRotated_L)

    # Rotating from Moon Synodic [M] to Synodic [S]
    controlAction_S = np.array([-rotatedControlAction[0], -rotatedControlAction[1], rotatedControlAction[2]])

    return controlAction_S