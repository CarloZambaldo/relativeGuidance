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

    # Computing relative state in Moon-centered Synodic and rotating to LVLH
    ## NO NOISE VERSION ##
        # relativeState_M = chaserState_M - targetState_M
        # relativeState_L, _ = convert_M_to_LVLH(targetState_M, relativeState_M, param)


    ################################## NOISE VERSION HERE #
    
    ## generation of navigation errors
    relativeState_L, _ = convert_M_to_LVLH(targetState_M, chaserState_M - targetState_M, param) # Only to compute the error ! this has to be updated after
    relativeState_L = inject_nav_error(relativeState_L, param)

    # Computing target state with disturbances (in LVLH frame)
    targetState_M = chaserState_M  - relativeState_L
    
    ### ############################################### END DISTURBANCES HERE #

    return targetState_M, chaserState_M, relativeState_L


def inject_nav_error(state, param):
    """
    Insert noise in the state vector. Considering a gaussian noise of: 3% on the position and 3% (component-wise) on the velocity.

    Adds a simple plateau to avoid unbounded growth of noise when far away:
    - cap position-based scaling using a max distance (default 10 km)
    - cap velocity-based scaling using a max speed per component (default 5 m/s)

    Thresholds can be overridden via:
    - param.nav_pos_plateau_m (meters)
    - param.nav_vel_plateau_ms (m/s)

    """

    val = 3/100
    
    r = state[:3]
    v = state[3:]

    # Plateau thresholds (dimensional)
    r_max_m = getattr(param, 'nav_pos_plateau_m', 10_000.0)  # 10 km
    v_max_ms = getattr(param, 'nav_vel_plateau_ms', 5.0)      # 5 m/s

    # Convert thresholds to nondimensional units if scales are available
    try:
        r_max_nd = r_max_m / param.xc
        v_max_nd = v_max_ms / (param.xc / param.tc)
    except Exception:
        # If scales are missing, fall back to no plateau
        r_max_nd = np.inf
        v_max_nd = np.inf

    # Position noise: std proportional to min(|r|, r_max_nd)
    r_scale = min(np.linalg.norm(r), r_max_nd)
    err_r = np.random.normal(0.0, val * r_scale, size=3)

    # Velocity noise: per-component std proportional to min(|v_i|, v_max_nd)
    v_scale = np.minimum(np.abs(v), v_max_nd)
    err_v = np.random.normal(0.0, val * v_scale, size=3)

    # print(f"err_r: {err_r}, r: {r*param.xc}")
    # print(f"err_v: {err_v}, v: {v*param.xc/param.tc}")

    return np.hstack([r + err_r, v + err_v])
