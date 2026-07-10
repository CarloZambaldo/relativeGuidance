import numpy as np
import time
from .ReferenceFrames import convert_M_to_LVLH

def OBNavigation(targetState_S, chaserState_S, previous_noise, param):
    """
    This function outputs the translation and rotation from Synodic to
    Moon-centered synodic and the relative state in LVLH

    Navigation errors are injected on the RELATIVE state only: the target
    ephemeris is assumed to be known on board (see thesis, Sec. 5.1), hence
    the returned targetState_M is the true one. The noisy relative state
    models the output of the on-board relative navigation filter.
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
    relativeState_L, _ = convert_M_to_LVLH(targetState_M, chaserState_M - targetState_M, param)

    # generation of navigation errors (on the relative state only)
    relativeState_L, newNoiseSample = inject_nav_error(relativeState_L, param, previous_noise)

    return targetState_M, chaserState_M, relativeState_L, newNoiseSample


def inject_nav_error(state, param, previous_noise=None):
    """
    Insert noise in the relative state vector, modelling the estimation error
    of the on-board relative navigation filter.

    The error on each channel is a first-order Gauss-Markov (exponentially
    correlated) process with bounded stationary standard deviation:

        err[k+1] = phi * err[k] + sqrt(1 - phi^2) * w[k],   phi = exp(-dt/tau)

    where w[k] ~ N(0, sigma^2). This keeps the error correlated in time
    (correlation time tau) WITHOUT the unbounded variance growth of a pure
    random walk, so the stationary std stays equal to sigma at every time.

    Standard deviations (1-sigma), with val = param.navigation_noise_percent:
    - position: sigma_r = val * min(||rho||, r_plateau)   (range-proportional,
      as for optical/lidar relative navigation, capped at plateau)
    - velocity: sigma_v = val * min(||v_rho||, v_plateau)

    Tunable via param (all have safe defaults):
    - param.navigation_noise_percent  e.g. 0.03 for 3% (None or 0 -> no noise)
    - param.nav_noise_corr_time_s     correlation time tau [s]      (default 60)
    - param.nav_pos_plateau_m         position scaling cap [m]      (default 10 km)
    - param.nav_vel_plateau_ms        velocity scaling cap [m/s]    (default 5 m/s)
    """

    val = getattr(param, 'navigation_noise_percent', None)
    if not val:  # None or 0.0 -> noiseless navigation (e.g. during training)
        return state, np.zeros(6)

    r = state[:3]
    v = state[3:]

    # Plateau thresholds (dimensional) converted to nondimensional units
    r_max_m = getattr(param, 'nav_pos_plateau_m', 10_000.0)   # 10 km
    v_max_ms = getattr(param, 'nav_vel_plateau_ms', 5.0)      # 5 m/s
    r_max_nd = r_max_m / (param.xc * 1e3)                     # xc is in km
    v_max_nd = v_max_ms / (param.xc * 1e3 / param.tc)         # xc/tc is in km/s

    # stationary standard deviations (isotropic, based on the norms)
    sigma_r = val * min(np.linalg.norm(r), r_max_nd)
    sigma_v = val * min(np.linalg.norm(v), v_max_nd)

    # Gauss-Markov propagation coefficient
    dt_s = param.tc / param.freqGNC                            # GNC step [s]
    tau_s = getattr(param, 'nav_noise_corr_time_s', 60.0)      # correlation time [s]
    phi = np.exp(-dt_s / tau_s)

    w_r = np.random.normal(0.0, sigma_r, size=3)
    w_v = np.random.normal(0.0, sigma_v, size=3)

    if previous_noise is not None:
        err_r = phi * previous_noise[:3] + np.sqrt(1 - phi**2) * w_r
        err_v = phi * previous_noise[3:] + np.sqrt(1 - phi**2) * w_v
    else:
        # initial error drawn from the stationary distribution
        err_r = w_r
        err_v = w_v

    newNoiseSample = np.hstack([err_r, err_v])

    return np.hstack([r + err_r, v + err_v]), newNoiseSample
