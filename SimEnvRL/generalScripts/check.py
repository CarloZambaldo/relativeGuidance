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
            if TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[2]**2 \
                  + TRUE_relativeState_S[3]**2 > (characteristicSize)**2:
                # the constraint is violated
                constraintViolationBool = True
                # violationEntity = 1-(np.sqrt(TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[2]**2 \
                #   + TRUE_relativeState_S[3]**2 - (characteristicSize)**2)/characteristicSize)

        case "CONE":
            currentRadius = TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[3]**2
            maxCurrentRadius = -(characteristicSize["acone"]**2*(TRUE_relativeState_S[2]-characteristicSize["bcone"])**3)
            
            if  currentRadius-maxCurrentRadius > 0:
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
                consider docked below 5 mm
        velocity:
                along R and H: max 0.04 m/s
                along V: max 0.1 m/s

    """

    crashedBool = False
    aimReachedBool = False

    match param.phaseID:
        case 1: # HOLDING STATE CONDITION
            if ( np.linalg.norm(TRUE_relativeState_L[0:3]-aimAtState[0:3]) <= 1.3007e-06 \
                and np.linalg.norm(TRUE_relativeState_L[3:6]-aimAtState[3:6]) <= 9.7737e-04 ):
                # under 500 m and 1 m/s
                aimReachedBool = True

        case 2: # DOCKING CONDITION
            crashedBool = False
            # if the position converges
            if (np.linalg.norm(TRUE_relativeState_L[0:3]-aimAtState[0:3]) <= 1.3007e-10):
                # stop when below 5 cm error
                # if also the velocity converges
                if (abs(TRUE_relativeState_L(4)-aimAtState(4)) <= 3.9095e-05 \
                    and abs(TRUE_relativeState_L(6)-aimAtState(6)) <= 3.9095e-05 \
                    and abs(TRUE_relativeState_L(5)-aimAtState(5)) <= 9.7737e-05 ):
                    aimReachedBool = True
                else:
                    crashedBool = True

        case _:
            pass

    return aimReachedBool, crashedBool