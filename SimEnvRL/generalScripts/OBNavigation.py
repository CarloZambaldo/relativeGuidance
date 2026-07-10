import numpy as np
import time
from .ReferenceFrames import convert_M_to_LVLH

def OBNavigation(targetState_S, chaserState_S, navMemory, param, appliedControl_L=None):
    """
    This function outputs the translation and rotation from Synodic to
    Moon-centered synodic and the relative state in LVLH

    Navigation errors are injected on the RELATIVE state only: the target
    ephemeris is assumed to be known on board (see thesis, Sec. 5.1), hence
    the returned targetState_M is the true one.

    The noisy measurement is then processed by a constant-gain navigation
    filter (steady-state Kalman-like): the previous estimate is propagated
    kinematically with the applied control acceleration and blended with the
    new measurement. This attenuates the high-frequency component of the
    measurement noise WITHOUT lagging the controlled dynamics (the control
    is fed forward in the prediction), avoiding SMC chattering on noise.

    navMemory: dict {'eps': normalized Gauss-Markov noise state,
                     'xhat': previous filtered estimate} or None at first call.
    appliedControl_L: control acceleration (LVLH, adimensional) applied over
                      the last GNC step, used for the kinematic prediction.
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
    # NOTE: newNoiseSample is the raw injected measurement error (for logging/analysis)
    eps_prev = navMemory.get('eps') if navMemory else None
    measurement_L, newNoiseSample, eps = inject_nav_error(relativeState_L, param, eps_prev)

    # constant-gain navigation filter (transparent when no noise is injected)
    val = getattr(param, 'navigation_noise_percent', None)
    Kf_r = getattr(param, 'nav_filter_gain_pos', 0.1)
    Kf_v = getattr(param, 'nav_filter_gain_vel', 0.1)
    xhat_prev = navMemory.get('xhat') if navMemory else None
    if (not val) or (xhat_prev is None):
        relativeState_L_est = measurement_L
    else:
        dt = 1.0 / param.freqGNC  # [adimensional] GNC step
        u = np.zeros(3) if appliedControl_L is None else np.asarray(appliedControl_L, dtype=float)
        # kinematic prediction of the previous estimate (control feed-forward;
        # the orbital relative accelerations are negligible over one GNC step)
        xpred = xhat_prev.copy()
        xpred[:3] += xhat_prev[3:] * dt + 0.5 * u * dt**2
        xpred[3:] += u * dt
        # measurement update with constant gains
        Kgain = np.hstack([Kf_r * np.ones(3), Kf_v * np.ones(3)])
        relativeState_L_est = xpred + Kgain * (measurement_L - xpred)

    newNavMemory = {'eps': eps, 'xhat': relativeState_L_est}

    return targetState_M, chaserState_M, relativeState_L_est, newNoiseSample, newNavMemory


def inject_nav_error(state, param, previous_noise=None):
    """
    Insert noise in the relative state vector, modelling the estimation error
    of the on-board relative navigation filter.

    The error on each channel is a first-order Gauss-Markov (exponentially
    correlated) process with bounded stationary standard deviation. The GM
    recursion runs on a NORMALIZED (unit-variance) state eps, which is then
    scaled by the current sigma:

        eps[k+1] = phi * eps[k] + sqrt(1 - phi^2) * w[k],   phi = exp(-dt/tau)
        err[k+1] = sigma[k+1] * eps[k+1],                   w[k] ~ N(0, 1)

    This keeps the error correlated in time (correlation time tau) WITHOUT the
    unbounded variance growth of a pure random walk, and makes the error track
    the current sigma instantly (the sensor accuracy improves as the range
    shrinks; without the rescaling the error would keep memory of the larger
    sigma from ~tau seconds earlier and spoil the terminal docking precision).

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
        return state, np.zeros(6), np.zeros(6)

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

    w = np.random.normal(0.0, 1.0, size=6)  # unit-variance white innovation

    if previous_noise is not None:
        eps = phi * previous_noise + np.sqrt(1 - phi**2) * w
    else:
        # initial normalized error drawn from the stationary distribution
        eps = w

    err_r = sigma_r * eps[:3]
    err_v = sigma_v * eps[3:]

    newNoiseSample = np.hstack([err_r, err_v])

    return np.hstack([r + err_r, v + err_v]), newNoiseSample, eps
