from . import ReferenceFrames
import numpy as np

def OBControl(targetState_S,uToBeRotated_L,param):
    """
    This function rotates the control action from LVLH to Synodic frame
    
    """

    # first clip the control action (+10% tol of the maxThrust)
    uToBeRotated_L = np.clip(uToBeRotated_L, -param.maxAdimThrust*1.1, param.maxAdimThrust*1.1)

    # get the target state in Moon Centered Synodic
    # translate from Synodic to Moon
    # Rotating from Moon to Moon Synodic [T14]
    targetState_M = np.array([-targetState_S[0] + (1-param.massRatio),
                              -targetState_S[1],
                              +targetState_S[2],
                              -targetState_S[3],
                              -targetState_S[4],
                              +targetState_S[5]])

    # Rotating matrices
    R_M_to_L, _ = ReferenceFrames.computeRotationMatrixLVLH(targetState_M, param)

    # Rotating from L to M
    rotatedControlAction = R_M_to_L.T @ uToBeRotated_L # NOTE: watch out for the transpose!

    # Rotating from Moon Synodic [M] to Synodic [S]
    controlAction_S = np.array([-rotatedControlAction[0],
                                -rotatedControlAction[1],
                                +rotatedControlAction[2]])

    return controlAction_S, uToBeRotated_L