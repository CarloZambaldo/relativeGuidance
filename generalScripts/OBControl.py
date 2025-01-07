from generalScripts import ReferenceFrames
import numpy as np

def OBControl(targetState_S,uToBeRotated_L,param):
    """
    This function rotates the control action from LVLH to Synodic frame
    
    """

    # get the target state in Moon Centered Synodic
    # translate from Synodic to Moon
    rM = np.array([1-param.massRatio,0,0]) # position of the moon in Synodic frame
    target_state_SCM = targetState_S - np.hstack([rM,[0,0,0]])

    # Rotating from Moon to Moon Synodic [T14]
    targetState_M = np.array([-target_state_SCM[0],-target_state_SCM[1],target_state_SCM[2],-target_state_SCM[3],-target_state_SCM[4],target_state_SCM[5]])

    # Rotating matrices
    R_M_to_L, _ = ReferenceFrames.computeRotationMatrixLVLH(targetState_M, param)

    # Rotating from L to M
    rotatedControlAction = R_M_to_L.T @ uToBeRotated_L # NOTE: watch out for the transpose!

    # Rotating from Moon Synodic [M] to Synodic [S]
    controlAction_S = np.array([-rotatedControlAction[0], -rotatedControlAction[1], rotatedControlAction[2]])

    return controlAction_S