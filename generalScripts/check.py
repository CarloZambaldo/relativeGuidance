import numpy as np

def constraintViolation(TRUE_relativeState_S,constraintType,characteristicSize,param):
    # translate to meters the relative state, since all the constraints are defined in meters
    TRUE_relativeState_S *= param.xc*1e3
    
    # default is no violation
    constraintViolationBool = False
    violationEntity = 0 # percentage of the violation (wrt characteristic size)

    # check collision with the constraints
    match constraintType:
        case "SPHERE":
            if TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[2]**2 \
                  + TRUE_relativeState_S[3]**2 > (characteristicSize)**2:
                # the constraint is violated
                constraintViolationBool = True
                violationEntity = 1-(np.sqrt(TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[2]**2 \
                  + TRUE_relativeState_S[3]**2 - (characteristicSize)**2)/characteristicSize)

        case "CONE":
            currentRadius = TRUE_relativeState_S[1]**2 + TRUE_relativeState_S[3]**2
            maxCurrentRadius = -(characteristicSize["acone"]**2*(TRUE_relativeState_S[2]-characteristicSize["bcone"])**3)
            
            if  currentRadius-maxCurrentRadius > 0:
                constraintViolationBool = True
                violationEntity = abs((currentRadius-maxCurrentRadius)/(maxCurrentRadius))
                violationEntity = max(violationEntity,10)
        case _:
            raise ValueError("Constraint Type not defined correctly")
        
    return constraintViolationBool, violationEntity

def aimReached(TRUE_relativeState_S, aimAtState, param):
    # check if the targetted aim is reached

    crashedBool = False
    # if the position converges
    if (np.linalg.norm(TRUE_relativeState_S[0:3]-aimAtState[0:3]) <= 1.3007e-10): # stop when below 5 cm error
        # if also the velocity converges
        if (np.linalg.norm(TRUE_relativeState_S[3:]-aimAtState[3:]) <= 1e-5): # stop when below 1cm/s
            aimReachedBool = True
        else:
            aimReachedBool = False
            crashedBool = True
    else:
        aimReachedBool = False

    return aimReachedBool, crashedBool