import numpy as np

def constraintViolation(TRUE_relativeState_L_meters,constraintType,characteristicSize,param):
    # default is no violation
    constraintViolationBool = False
    violationEntity = 0 # percentage of the violation (wrt characteristic size)

    # check collision with the constraints
    match constraintType:
        case "SPHERE":
            currentRadius2 = TRUE_relativeState_L_meters[0]**2 + TRUE_relativeState_L_meters[1]**2 + TRUE_relativeState_L_meters[2]**2
            maxCurrentRadius2 = characteristicSize**2
            if currentRadius2 <= maxCurrentRadius2:
                # the constraint is violated
                constraintViolationBool = True

            # compute the relative position (normalized to 1 on the constraint)
            violationEntity = 1 - (currentRadius2 - maxCurrentRadius2)/maxCurrentRadius2

        case "CONE":
            currentRadius2 = TRUE_relativeState_L_meters[0]**2 + TRUE_relativeState_L_meters[2]**2 # for a given V-BAR the cone is sliced on R-H plane
            maxCurrentRadius2 = characteristicSize["acone"]**2 * (TRUE_relativeState_L_meters[1] - characteristicSize["bcone"])**3
            
            if  currentRadius2 + maxCurrentRadius2 > 0:
                constraintViolationBool = True
                # violationEntity = max(violationEntity,10)

            # compute the relative position (normalized to 1 on the constraint)
            # TODO: for a future me: check if this equation still holds when rho_V_bar > 0
            violationEntity = (currentRadius2 + maxCurrentRadius2)/abs(maxCurrentRadius2)
        case _:
            raise ValueError("Constraint Type not defined correctly")
        
    return constraintViolationBool, violationEntity



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
            if np.linalg.norm(TRUE_relativeState_L[0:3]-aimAtState[0:3]) < 2.6014e-08 \
               and np.linalg.norm(TRUE_relativeState_L[3:]-aimAtState[3:]) < 4.8868e-04:
                # under 10 m tollerance and 0.5 m/s
                aimReachedBool = True
                # note that it is not possible to crash in phase 1... (the chaser is distant from the target!)
        case 2:
            if (TRUE_relativeState_L[1] - aimAtState[1] >= -1.3007e-11):  # when below 5 mm along V-BAR (from -5mm to in front of the target)
                # check if converged on R and H:
                if (np.linalg.norm(TRUE_relativeState_L[[0,2]]-aimAtState[[0,2]]) <= 1.3007e-10):  # when below 5 cm error (5cm = 1.3007e-10)
                    # if also the velocity converges
                    # docking standard: along R and H: max 0.04 m/s; along V: max 0.1 m/s
                    if (abs(TRUE_relativeState_L[3]-aimAtState[3]) <= 3.9095e-05 and \
                        abs(TRUE_relativeState_L[5]-aimAtState[5]) <= 3.9095e-05 and \
                        abs(TRUE_relativeState_L[4]-aimAtState[4]) <= 9.7737e-05):
                        aimReachedBool = True
                    else:
                        crashedBool = True
                else:
                    crashedBool = True
            elif (TRUE_relativeState_L[1] - aimAtState[1]) > 0: # if in front of the target
                # if the velocity has not converget yet
                if not (abs(TRUE_relativeState_L[3]-aimAtState[3]) <= 3.9095e-05 and \
                    abs(TRUE_relativeState_L[5]-aimAtState[5]) <= 3.9095e-05 and \
                    abs(TRUE_relativeState_L[4]-aimAtState[4]) <= 9.7737e-05) or \
                    not (np.linalg.norm(TRUE_relativeState_L[[0,2]]-aimAtState[[0,2]]) <= 1.3007e-10):
                    crashedBool = True
        case _:
            raise ValueError("The termination condition for the given phaseID has not been implemented yet.")

    return aimReachedBool, crashedBool




## CHECK CONTRAINTS VIOLATION ##################################################################################
def OBoTviolations(OBoptimalTrajectory, constraintType, characteristicSize):
    violationFlag = False
    violationPosition = []
    
    # if OBoptimalTrajectory and ('state' in OBoptimalTrajectory):
    #     trajectory = OBoptimalTrajectory['state']
    #     if 'controlAction' in OBoptimalTrajectory:
    #         controlAction = OBoptimalTrajectory['controlAction']
    #         for idx, control in enumerate(controlAction):
    #             if np.linalg.norm(control) > 12:
    #                 violationFlag = True
    #                 violationPosition.append((2, idx))
    #                 # warning("Violation of Thrust Constraint")
    #     else:
    #         violationPosition = [(1, idx) for idx in range(trajectory.shape[1])]
    # else:
    #     return violationFlag, violationPosition
    if OBoptimalTrajectory and ('state' in OBoptimalTrajectory):
        trajectory = OBoptimalTrajectory['state']
        match constraintType:
            case 'SPHERE':
                for idx in range(trajectory.shape[1]):
                    if np.sum(trajectory[:3, idx]**2) <= characteristicSize**2:
                        violationFlag = True
                        violationPosition.append((1, idx))

            case 'CONE':
                for idx in range(trajectory.shape[1]):
                    if (characteristicSize['acone']**2 * (trajectory[1, idx] - characteristicSize['bcone'])**3 +
                        trajectory[0, idx]**2 + trajectory[2, idx]**2) > 0:
                        violationFlag = True
                        violationPosition.append((1, idx))

        # if violationFlag:
        #     print("Warning: The computed Trajectory violates the constraints.")
        # else:
        #     print("No violations of the constraints identified.")

    return violationFlag, violationPosition