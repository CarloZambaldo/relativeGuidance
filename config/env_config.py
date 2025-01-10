## Configuration for the Physical Environment ##

from dataclasses import dataclass, field
from generalScripts.ReferenceFrames import convert_S_to_LVLH, convert_LVLH_to_S
import numpy as np
import scipy.io
import random

@dataclass(frozen=True)
class physParamClass:
    # ENVIRONMENT PARAMETERS #
    xc : float = 384400              # units for adimensional space [km]
    tc : float = 1/(2.661699e-6)  # units for adimensional time [s]
    massEarth = 5.973698863559727e+24 # [kg]
    massMoon  = 7.347673092457352e+22 # [kg]
    massRatio : float = massMoon/(massEarth+massMoon)
    Omega : float = 2*np.pi/2358720    # [rad/s]
    SolarFlux : float = 1361/299792458 # [W/m^2 / (m/s)] Solar Flux at 1 AU

    # SIMULATION PARAMETERS #
    tspan = np.array([0, 0.008])                            # initial and final time for simulation    [ADIMENSIONAL]
    phaseID = 1

    maxAdimThrust : float = (490/15000)*1e-3/xc*tc**2         # maximum adimensional acceleration [adimensional]
    holdingState = np.array([0, -8/xc, 0, 0, 0, 0])          # [adimensional]
    dockingState = np.array([0, 0, 0, 0, 0.06e-3*tc/xc, 0]) # Final relative state from Luca Thesis
    freqGNC : float = 10 * tc                                # [adimensional Hz] GNC upadate frequency

    # SPACECRAFT PARAMETERS #
    chaser: dict = field(default_factory=lambda: {
        "mass": 15000,                  # [kg]
        "Area": 18,                     # [m^2]
        "reflCoeffSpecular": 0.5,       # [-]
        "reflCoeffDiffuse":  0.1        # [-]
    })

    target: dict = field(default_factory=lambda: {
        "mass": 40000,                      # [kg]
        "Area": 110,                      # [m^2]
        "reflCoeffSpecular": 0.5,        # [-]
        "reflCoeffDiffuse":  0.1        # [-]
    })


#### define the initial values ####
@dataclass()
class initialValueClass():
    targetState_S : np.ndarray = field(default_factory=lambda: np.zeros(6,))
    chaserState_S : np.ndarray = field(default_factory=lambda: np.zeros(6,))
    DeltaIC_S : np.ndarray = field(default_factory=lambda: np.zeros(6,))
    relativeState_L : np.ndarray = field(default_factory=lambda: np.zeros(6,))
    seedValue : int = None

    def define_initialValues(self,param,seed=None):
        # default initial conditions for target state
        #targetState_S = np.array([1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]) # this is obtained from PhD thesis 

        # setting random seed
        self.seedValue = random.seed(seed)

        # Extract the 'refTraj' structured array from the refTraj.mat file
        mat_data  = scipy.io.loadmat(r"./config/refTraj.mat")
        refTraj = mat_data['refTraj']
        # Access the trajectory within the structured array
        referenceStates = refTraj['y'][0, 0]        # Main trajectory data
        # Extract a random position
        rndmnbr = random.randint(1,np.size(referenceStates,1)) # random position inside the reference trajectory
        targetState_S = referenceStates[:,rndmnbr]

        match param.phaseID:
            case 1: # Apporaching Safe Holding Point
                # defining the random initial relative state (IN SYNODYC!)
                condSodd = False
                tentativi = 0
                while not condSodd:
                    rand_position = (-8+16*np.random.rand(3))          # position range [-8,+8] km
                    # print(f"Random Position Generated: {rand_position}\n")
                    if np.linalg.norm(rand_position) > .2: # position must be greater than 200 meters
                        condSodd = True
                        rand_position /= param.xc # adimensionalize
                    tentativi += 1        
                    if tentativi > 50:
                        raise Exception("Maximum number of attempts reached for random initial conditions definition: the computed state violates the Keep Out Sphere.\nTry again with different seed value.")
                rand_velocity = (-5+10*np.random.rand(3)) * 1e-3 / param.xc * param.tc # velocity range [-5,+5] m/s
                DeltaIC_S = np.hstack([rand_position, rand_velocity])

            case 2: # Docking
                # defining the random initial relative state (IN LVLH!)
                rand_position_L = np.array([(-8+6*np.random.rand()),(-8+6*np.random.rand()),(-8+6*np.random.rand())]) / param.xc          # position range along V-BAR [-8,-2] km
                rand_velocity_L = (-5+10*np.random.rand(3)) * 1e-3 / param.xc * param.tc # velocity range
                
                DeltaIC_S = convert_LVLH_to_S(targetState_S,np.hstack([rand_position_L, rand_velocity_L]),param)
                ## DEBUG: test_prima = np.hstack([rand_position_L, rand_velocity_L])
                ## DEBUG: test_dopo = convert_S_to_LVLH(targetState_S, DeltaIC_S, param)
                ## DEBUG: if abs(np.linalg.norm(DeltaIC_S)-np.linalg.norm(np.hstack([rand_position_L, rand_velocity_L]))) > 1e-8 \
                ## DEBUG:    or abs(np.linalg.norm(test_dopo-test_prima)) > 1e-8:
                ## DEBUG:     raise ArithmeticError("Rotation of the initial condition led to a wrong result")

            case _:
                raise Exception("PhaseID not recognized. Please select a valid phaseID.")
        
        ########### ## BUGFIXING CODE ##
        ########### targetState_S = np.array([ 1.01056035, -0.03617079, -0.13739958, -0.05186586, -0.05896856, 0.22699674])
        ########### chaserState_S = np.array([ 1.01054142, -0.03616771, -0.13740013, -0.0486426 , -0.05719729, 0.22417574])
        ########### DeltaIC_S = chaserState_S-targetState_S


        # computing the relative state in LVLH
        chaserState_S = targetState_S + DeltaIC_S
        relativeState_L = convert_S_to_LVLH(targetState_S, DeltaIC_S, param)

        # assigning the values to the class
        self.targetState_S = targetState_S
        self.chaserState_S = chaserState_S
        self.DeltaIC_S = DeltaIC_S
        self.relativeState_L = relativeState_L
        self.fullInitialState = np.hstack([targetState_S, chaserState_S])

        print("\n==============================================================\n")
        print(f"Correctly defined initial conditions:")

        # Print initial distance and velocity between C and T
        initial_distance = np.linalg.norm(DeltaIC_S[:3]) * param.xc
        initial_velocity = np.linalg.norm(DeltaIC_S[3:]) * param.xc * 1e3 / param.tc
        
        print(f"  Initial Distance between C and T: {initial_distance:.2f} [km]")
        print(f"  Initial Relative velocity between C and T: {initial_velocity:.2f} [m/s]\n")
        print(f" [ seed =",self.seedValue,"]")
        print("==============================================================\n")
        return self


# defining the parameters
def get(seed=None):
    param = physParamClass()
    initialValue = initialValueClass()
    initialValue = initialValue.define_initialValues(param,seed)

    return param, initialValue