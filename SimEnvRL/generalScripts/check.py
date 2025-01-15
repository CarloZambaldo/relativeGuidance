import numpy as np

def constraintViolation(TRUE_relativeState_S,constraintType,characteristicSize,param):
    # translate to meters the relative state, since all the constraints are defined in meters
    TRUE_relativeState_S *= param.xc*1e3
    
    # default is no violation
    constraintViolationBool = False
    #violationEntity = 0 # percentage of the violation (wrt characteristic size)

    # check collision with the constraints
    match constraintType:
        case "SPHERE":
            if TRUE_relativeState_S[0]**2 + TRUE_relativeState_S[1]**2 \
                  + TRUE_relativeState_S[2]**2 > (characteristicSize)**2:
                # the constraint is violated
                constraintViolationBool = True
                # violationEntity = 1-(np.sqrt(TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[2]**2 \
                #   + TRUE_relativeState_S[3]**2 - (characteristicSize)**2)/characteristicSize)

        case "CONE":
            currentRadius = TRUE_relativeState_S[0]**2 + TRUE_relativeState_S[2]**2
            maxCurrentRadius = -(characteristicSize["acone"]**2*(TRUE_relativeState_S[1]-characteristicSize["bcone"])**3)
            
            if  currentRadius - maxCurrentRadius > 0:
                constraintViolationBool = True
                # violationEntity = abs((currentRadius-maxCurrentRadius)/(maxCurrentRadius))
                # violationEntity = max(violationEntity,10)
        case _:
            raise ValueError("Constraint Type not defined correctly")
        
    return constraintViolationBool #, violationEntity



##ok
def aimReached(TRUE_relativeState_L, aimAtState, param):
    """
    check if the targetted aim is reached

    consider the folowing constraints for phase 1:
        position: 200 m
        velocity: 0.5 m/s
        
    consider the docking standards for phase 2: 
        position:
                consider docked below 5 cm
        velocity:
                along R and H: max 0.04 m/s
                along V: max 0.1 m/s

    """
    crashedBool = False
    aimReachedBool = False

    # if the position converges
    match param.phaseID:
        case 1:
            if np.linalg.norm(TRUE_relativeState_L[0:3]-aimAtState[0:3]) < 5.2029e-07 \
               and np.linalg.norm(TRUE_relativeState_L[3:]-aimAtState[3:]) < 4.8868e-04:
                # under 200 m tollerance and 0.5 m/s
                aimReachedBool = True
                # note that it is not possible to crash in phase 1... (the chaser is distant from the target!)
        case 2:
            if (np.linalg.norm(TRUE_relativeState_L[1]-aimAtState[1]) <= 1.3007e-10):  # when below 5 cm along V-BAR check if converged:
                if (np.linalg.norm(TRUE_relativeState_L[[0, 2]]-aimAtState[[0, 2]]) <= 2.6015e-10):  # stop when below 10 cm error (5cm = 1.3007e-10)
                    # if also the velocity converges
                    # docking standard: along R and H: max 0.04 m/s; along V: max 0.1 m/s
                    if (abs(TRUE_relativeState_L[3]-aimAtState[3]) <= 3.9095e-05 and
                        abs(TRUE_relativeState_L[5]-aimAtState[5]) <= 3.9095e-05 and
                        abs(TRUE_relativeState_L[4]-aimAtState[4]) <= 9.7737e-05):
                        aimReachedBool = True
                    else:
                        crashedBool = True
                elif (TRUE_relativeState_L[1]-aimAtState[1]) > 0:  # stop when in front of the target
                    crashedBool = True

        case _:
            raise ValueError("The termination condition for the given phaseID has not been implemented yet.")

    return aimReachedBool, crashedBool