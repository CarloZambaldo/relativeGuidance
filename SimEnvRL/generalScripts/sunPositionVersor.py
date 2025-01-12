import numpy as np

def sunPositionVersor(t, param):
    """
    Computes the unit vector pointing towards the Sun, based on the time and the initial angle.
    NOTE: here it is assumed that the sun orbits the Earth-Moon system in a circular orbit.
    
    Parameters:
        t (float): Current time.
        param (Param): Instance of the Param dataclass, containing all necessary parameters.
    
    Returns:
        sunVersor (np.array): Unit vector pointing towards the Sun.
    """
    
    # Set the initial angle to zero if not provided
    if not hasattr(param, 'sunInitialAngle'):
        param.sunInitialAngle = 0.0 # [rad]
    
    # Calculate the angle  [rad]
    theta = param.sunInitialAngle + 1.996437750711854e-07*t
    
    # Compute the Sun's direction unit vector
    sun_versor = np.array([np.cos(theta), np.sin(theta), 0])
    
    return sun_versor