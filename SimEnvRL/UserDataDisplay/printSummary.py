from ..generalScripts import *
import numpy as np

def printSummary(env):
    param = env.param
    initialValue = env.initialValue
    adimensionalSolution = {
        "x": env.timeHistory.T,
        "y": env.fullStateHistory.T,
        "controlAction": env.controlActionHistory_L.T
    }
    OBoptimalTrajectory = env.OBoptimalTrajectory
    # Calculations
    _, _, relativeState_L = OBNavigation(initialValue.targetState_S, initialValue.chaserState_S, param)
    if relativeState_L[1] <= 0:
        rP = "BEHIND"
    else:
        rP = "IN FRONT OF"

    gg = (relativeState_L[3:6] @ relativeState_L[0:3]) / (np.linalg.norm(relativeState_L[3:6]) * np.linalg.norm(relativeState_L[0:3]))
    if gg > 0:
        rV = "AWAY FROM"
    else:
        rV = "TOWARDS"
    anglo = np.arccos(gg)

    relDynami = np.zeros((len(adimensionalSolution["x"]), 6))
    soluz = np.array(adimensionalSolution["y"]).T
    for idx in range(len(adimensionalSolution["x"])):
        rotatedRelativeState = ReferenceFrames.convert_S_to_LVLH(soluz[idx, 0:6], soluz[idx, 6:12] - soluz[idx, 0:6], param)
        relDynami[idx, 0:6] = rotatedRelativeState

    controlAction = adimensionalSolution["controlAction"]
    relDynami = relDynami * param.xc
    if param.phaseID == 1:
        positionError = np.linalg.norm(relDynami[-1, 0:3] - param.holdingState[0:3] * param.xc) * 1e3
        velocityError = np.linalg.norm(relDynami[-1, 3:6] - param.holdingState[3:6] * param.xc) * 1e3 / param.tc
    elif param.phaseID == 2:
        positionError = np.linalg.norm(relDynami[-1, 0:3] - param.dockingState[0:3] * param.xc) * 1e3
        velocityError = np.linalg.norm(relDynami[-1, 3:6] - param.dockingState[3:6] * param.xc) * 1e3 / param.tc

    # Prints
    print("#############################")

    print(f"\nPhaseID: {param.phaseID}")
    print(f"Seed imposed: ",initialValue.seedValue,"")
    print(f"Simulated Time: {param.tspan[1] / 3600 * param.tc:.2f} [hours]")
    if OBoptimalTrajectory:
        print(f"Optimal Trajectory TOF estimated: {OBoptimalTrajectory.t[-1] / 3600 * param.tc:.2f} [hours]")
    print(f"Initial Distance between C and T: {np.linalg.norm(initialValue.DeltaIC_S[0:3]) * param.xc:.2f} [km]")
    print(f"Initial Relative velocity between C and T: {np.linalg.norm(initialValue.DeltaIC_S[3:6]) * param.xc / param.tc * 1e3:.2f} [m/s]")
    print(f"Initially: Chaser is *{rP}* the Target and moving *{rV}* the Target at an angle of {np.rad2deg(anglo):.2f} [deg]")

    print("\n#############################\n")

    print(f"Maximum Thrust Available: {param.maxAdimThrust} [-]")
    print(f"Maximum Thrust Required : {np.max(controlAction):g} [-]")
    print(f"Maximum Thrust Required (norm) : {np.max(np.linalg.norm(controlAction, axis=1)):g} [-]")

    print(f"Actual Final Position Error:    {positionError:g} [m]")
    print(f"Actual Final Velocity Error:    {velocityError:g} [m/s]")

    print("\n#################################")
    print("############## END ##############")
    print("#################################")
