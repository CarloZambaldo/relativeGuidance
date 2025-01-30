## Configuration for the Physical Environment ##

from dataclasses import dataclass, field
from ..generalScripts.ReferenceFrames import convert_S_to_LVLH, convert_LVLH_to_S
import numpy as np
import scipy.io
#import random

@dataclass(frozen=True)
class physParamClass:
    # TUNABLE PARAMETERS
    phaseID : int = None                                     # default value set to none
    tspan : np.ndarray = field(default_factory=lambda: None) # default value initial and final time for simulation    [ADIMENSIONAL]

    # ENVIRONMENT PARAMETERS #
    xc : float = 384400.                                     # units for adimensional space [km]
    tc : float = 1/(2.661699e-6)                             # units for adimensional time [s]
    massEarth = 5.973698863559727e+24                        # [kg]
    massMoon  = 7.347673092457352e+22                        # [kg]
    massRatio : float = massMoon/(massEarth+massMoon)        # mass ratio for for the given CR3BP
    Omega : float = 2*np.pi/2358720                          # [rad/s]
    SolarFlux : float = 1361/299792458                       # [W/m^2 / (m/s)] Solar Flux at 1 AU
    sunInitialAngle : float = 2 * np.pi * np.random.rand()   # random initial angle of the sun

    # SIMULATION PARAMETERS #
    maxAdimThrust : float = (490/15000)*1e-3/xc*tc**2        # maximum adimensional acceleration [adimensional]
    specificImpulse: float = 270                             # [s] specific impulse of the thruster 
    holdingState = np.array([0, -4/xc, 0, 0, 0, 0])          # [adimensional]
    dockingState = np.array([0, 0, 0, 0, 0.02e-3*tc/xc, 0])  # Final relative state similar to Luca Thesis
    freqGNC : float = 5 * tc                                 # [adimensional (from Hz)] GNC upadate frequency
    RLGNCratio : int = 100                                   # number of GNC steps per RL step

    # SPACECRAFT PARAMETERS #
    chaser: dict = field(default_factory=lambda: {
        "mass": 15000.,                 # [kg]
        "Area": 18.,                    # [m^2]
        "reflCoeffSpecular": 0.5,       # [-]
        "reflCoeffDiffuse":  0.1        # [-]
    })

    target: dict = field(default_factory=lambda: {
        "mass": 40000.,                 # [kg]
        "Area": 110.,                   # [m^2]
        "reflCoeffSpecular": 0.5,       # [-]
        "reflCoeffDiffuse":  0.1        # [-]
    })

    constraint : dict = None

    # SETTING DEFAULTS
    def __post_init__(self):
        # Initialize tc if not provided (could also compute other variables)
        if self.phaseID is None:
            object.__setattr__(self, 'phaseID', 2)
        if self.tspan is None:
            object.__setattr__(self, 'tspan', np.array([0, 0.02]))

        # ENVIRONMENT CONSTRAINTS
        match self.phaseID:
            case 1:
                object.__setattr__(self,'constraint',
                                    {
                                        "constraintType" : 'SPHERE',
                                        "aimAtState" : self.holdingState,
                                        "characteristicSize" : 200 
                                    }
                )
            case 2:
                object.__setattr__(self,'constraint',
                                    {
                                        "constraintType" : 'CONE',
                                        "aimAtState" : self.dockingState,
                                        "characteristicSize" : {'acone': 0.02, 'bcone': 10}
                                    }
                )
            case _:
                raise ValueError("Phase ID not defined correctly")
            

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
        self.seedValue = np.random.seed(seed)

        # Extract the 'refTraj' structured array from the refTraj.mat file
        mat_data  = scipy.io.loadmat(r"SimEnvRL/config/refTraj.mat")
        refTraj = mat_data['refTraj']
        # Access the trajectory within the structured array
        referenceStates = refTraj['y'][0, 0]        # Main trajectory data
        # Extract a random position
        rndmnbr = np.random.randint(1,np.size(referenceStates,1)) # random position inside the reference trajectory
        targetState_S = referenceStates[:,rndmnbr]

        match param.phaseID:
            case 1: # Apporaching Safe Holding Point #########################################
                # defining the random initial relative state (IN SYNODYC!)
                tentativi = 100
                while tentativi > 0:
                    rand_position_S = (-9+18*np.random.rand(3))          # position range [-9,+9] km
                    # print(f"Random Position Generated: {rand_position_S}\n")
                    if np.linalg.norm(rand_position_S) > .2: # position must be greater than 200 meters
                        rand_position_S /= param.xc # adimensionalize
                        tentativi = 0
                    tentativi -= 1

                if tentativi != -1:
                    raise Exception("Maximum number of attempts reached for random initial conditions definition: the computed state violates the Keep Out Sphere.\nTry again with different seed value.")
                
                rand_velocity_S = (-5+10*np.random.rand(3)) * 1e-3 / param.xc * param.tc # velocity range [-5,+5] m/s
                DeltaIC_S = np.hstack([rand_position_S, rand_velocity_S])

            case 2: # DOCKING #########################################
                # defining the random initial relative state (IN LVLH!)
                rand_position_L = np.array([(-2+4*np.random.rand()),              # R-BAR # position range along R-BAR [-2,+2] km
                                            (-8+6.5*np.random.rand()),            # V-BAR # position range along V-BAR [-8,-1.5] km
                                            (-2+4*np.random.rand())]) / param.xc  # H-BAR # position range along H-BAR [-2,+2] km
                rand_velocity_L = (-2+2*np.random.rand(3)) * 1e-3 / param.xc * param.tc    # velocity range
                
                DeltaIC_S = convert_LVLH_to_S(targetState_S,np.hstack([rand_position_L, rand_velocity_L]),param)

            case _:
                raise Exception("PhaseID not recognized. Please select a valid phaseID.")
        
        # computing the relative state in LVLH
        chaserState_S = targetState_S + DeltaIC_S
        relativeState_L = convert_S_to_LVLH(targetState_S, DeltaIC_S, param)

        # assigning the values to the class
        self.targetState_S = targetState_S
        self.chaserState_S = chaserState_S
        self.DeltaIC_S = DeltaIC_S
        self.relativeState_L = relativeState_L
        self.fullInitialState = np.hstack([targetState_S, chaserState_S])

        self.printIC(param)

        return self
    
    def imporse_initialValues(self,param,values):
        # Check if either 'targetState_S' or 'relativeState_L' is None or not a numpy array
        if values["targetState_S"] is None \
           or values["relativeState_L"] is None \
           or not isinstance(values["targetState_S"], np.ndarray) \
           or not isinstance(values["relativeState_L"], np.ndarray) \
           or values["targetState_S"].size != 6 \
           or values["relativeState_L"].size != 6:
            raise RuntimeError("COULD NOT RETRIEVE THE INITIAL CONDITIONS")
        
        targetState_S = values["targetState_S"].flatten()
        relativeState_L = values["relativeState_L"].flatten()

        DeltaIC_S = convert_LVLH_to_S(targetState_S,relativeState_L,param)
        chaserState_S = targetState_S + DeltaIC_S
        
        self.targetState_S = targetState_S
        self.chaserState_S = chaserState_S
        self.DeltaIC_S = DeltaIC_S
        self.relativeState_L = relativeState_L
        self.fullInitialState = np.hstack([targetState_S, chaserState_S])

        print(f"CORRECTLY IMPOSED INITIAL CONDITIONS. relativeState_L (adim) = {relativeState_L}")
        self.printIC(param)

        return self
    
    def printIC(self,param):
        print("\n==============================================================")
        print(f"Correctly defined initial conditions:")

        # Print initial distance and velocity between C and T
        initial_distance = np.linalg.norm(self.DeltaIC_S[:3]) * param.xc
        initial_velocity = np.linalg.norm(self.DeltaIC_S[3:]) * param.xc * 1e3 / param.tc
        
        print(f"   Initial Distance between C and T: {initial_distance:.2f} [km]")
        print(f"   Initial Relative velocity between C and T: {initial_velocity:.2f} [m/s]")

        # compute the target position "descriptor"
        angol = (np.arctan2(self.targetState_S[2],self.targetState_S[1]) *180/np.pi)
        if angol > -180 and angol <= 90:
            angol = - angol - 90
        elif angol > 90 and angol <= 180:
            angol = + angol + 90
        posiz = "N/A"
        if angol <= 30 and angol >= -30:
            posiz = "APOSELENE"
        elif (angol > 30 and angol <= 90):
            posiz = "INTERMEDIATE - LEAVING APOSELENE"
        elif (angol < -30 and angol >= -90):
            posiz = "INTERMEDIATE - APPROACHING APOSELENE"
        elif (angol > 90 and angol <= 180) or (angol < -90 and angol >= -180):
            posiz = "PERISELENE"
        print(f"   TARGET POSITION: {posiz} (angle: {angol:.2f} deg)")
        print(f" [ seed =",self.seedValue,"]")
        print("==============================================================\n")



# defining the parameters for the environment
def getParam(phaseID=None,tspan=None):
    # define the environmental parameters (constant for the environment)
    param = physParamClass(phaseID=phaseID,tspan=tspan)

    return param

def getInitialValues(param,seed=None,values=None):
    # define the dataclass (with no values)
    initialValue = initialValueClass()

    # fill the class with the values passed by the used
    values = values or {}
    if values: # if the initial conditions are passed define accordingly the initialValue parameter
        values = {
            "targetState_S": None if "targetState_S" not in values or values["targetState_S"] is None or not isinstance(values["targetState_S"], np.ndarray) else values["targetState_S"],
            "relativeState_L": None if "relativeState_L" not in values or values["relativeState_L"] is None or not isinstance(values["relativeState_L"], np.ndarray) else values["relativeState_L"]
        }
        initialValue = initialValue.imporse_initialValues(param,values)
        typeOfInitialConditions = "USER_DEFINED"

    else: # otherwise use defaults
        initialValue = initialValue.define_initialValues(param,seed)
        typeOfInitialConditions = "DEFAULT"

    return initialValue, typeOfInitialConditions