import numpy as np
import time
from .ReferenceFrames import convert_M_to_LVLH

def OBNavigation(targetState_S, chaserState_S, param):
    """
    This function outputs the translation and rotation from Synodic to
    Moon-centered synodic and the relative state in LVLH

    """

    # Translating and rotating to Moon-centered Synodic [ex FranziRot]
    targetState_M = np.array([
        -targetState_S[0] + (1 - param.massRatio),
        -targetState_S[1],
        targetState_S[2],
        -targetState_S[3],
        -targetState_S[4],
        targetState_S[5]
    ])

    chaserState_M = np.array([
        -chaserState_S[0] + (1 - param.massRatio),
        -chaserState_S[1],
        chaserState_S[2],
        -chaserState_S[3],
        -chaserState_S[4],
        chaserState_S[5]
    ])
	

    ## ADD DISTURBANCES HERE #
    ## generation of navigation errors
    targetState_M = inject_nav_error(targetState_M, param)
    chaserState_M = inject_nav_error(chaserState_M, param)


    # Computing relative state in Moon-centered Synodic and rotating to LVLH
    relativeState_M = chaserState_M - targetState_M
    relativeState_L, _ = convert_M_to_LVLH(targetState_M, relativeState_M, param)

    
    return targetState_M, chaserState_M, relativeState_L

def inject_nav_error(state, param):
    """
    Insert noise in the state vector. Considering a gaussian noise of: 3% on the position and 3% (component-wise) on the velocity.

    """

    val = 0.03
    
    r = state[:3]
    v = state[3:]

    err_r = np.random.normal(0.0, np.sqrt(val*np.linalg.norm(r)), 3)
    err_v = np.sqrt( v * np.random.normal(0.0, 0.03, size=3) )

    # print(f"err_r: {err_r}, r: {r*param.xc}")
    # print(f"err_v: {err_v}, v: {v*param.xc/param.tc}")

    return np.hstack([r + err_r, v + err_v])