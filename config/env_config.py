## Configuration for the Physical Environment ##

from dataclasses import dataclass, field
from generalScripts.ReferenceFrames import rotate_S_to_LVLH
import numpy as np
import scipy.io
import random

@dataclass(frozen=True)
class physParamClass:
	xc : float = 384400		      # units for adimensional space [km]
	tc : float = 1/(2.661699e-6)  # units for adimensional time [s]

	massEarth = 5.973698863559727e+24 # [kg]
	massMoon  = 7.347673092457352e+22 # [kg]
	massRatio : float = massMoon/(massEarth+massMoon)
	Omega : float = 2*np.pi/2358720 # [rad/s]

@dataclass(frozen=True)
class initialValueClass():
	targetState_S : np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
	chaserState_S : np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
	DeltaIC_S : np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
	relativeState_L : np.ndarray = field(default_factory=lambda: np.zeros((6, 1)))
	
	def define_initialValues(self,param):
		## DETERMINISTIC
		#targetState_S = np.array([1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]) # this is obtained from PhD thesis 
		#DeltaIC_S = np.array([0,0,0,0,0,0])
		#chaserState_S = targetState_S + DeltaIC_S
		#relativeState_S = chaserState_S-targetState_S
		#relativeState_L = rotate_S_to_M(relativeState_S,)

		## RANDOM
		seedValue = random.seed()
		print(f"Using random initial conditions definition\n")
		print(f" [seed =",seedValue,"]")

		# Extract the 'refTraj' structured array from the refTraj.mat file
		mat_data  = scipy.io.loadmat('./refTraj.mat')
		refTraj = mat_data['refTraj']
		# Access the trajectory within the structured array
		referenceStates = refTraj['y'][0, 0]        # Main trajectory data
		# Extract a random position
		rndmnbr = random.randint(1,np.size(referenceStates,1)) # random position inside the reference trajectory
		targetState_S = referenceStates[:,rndmnbr]

		# defining the random initial relative distance
		deltaR = np.random.ranf(3)
		deltaR = 5/param.xc*deltaR/np.linalg.norm(deltaR) # maximum 5 km distance
		# defining the random initial relative velocity 
		deltaV = 0*np.random.ranf(3)

		# stacking and defining the initial Delta IC
		DeltaIC_S = np.hstack([deltaR, deltaV])
		chaserState_S = targetState_S + DeltaIC_S

		relativeState_L = rotate_S_to_LVLH(targetState_S, DeltaIC_S, param)

		return targetState_S, chaserState_S, DeltaIC_S, relativeState_L
	
	# function to check the relative state consistency
	def check_relative_state(self,param):
		# Compute the norm of the difference
		error_norm = np.linalg.norm(self.DeltaIC_S - self.DeltaIC_S)
		DeltaIC_S = self.DeltaIC_S
		if error_norm >= 1e-10:
			print(f"value for DeltaIC_S: {DeltaIC_S} (norm: {np.linalg.norm(DeltaIC_S)})")
			print(f"initialRelativeState: {self.DeltaIC_S} (norm: {np.linalg.norm(self.DeltaIC_S)})")
			print(f"Error norm: {error_norm}")
			raise ValueError("SOMETHING WENT WRONG IN THE COMPUTATION OF THE RELATIVE STATE!")
		else:
			print(f"Correctly defined initial conditions:")

		# Print initial distance and velocity between C and T
		initial_distance = np.linalg.norm(DeltaIC_S[:3]) * param.xc
		initial_velocity = np.linalg.norm(DeltaIC_S[3:]) * param.xc / param.tc
		
		print(f"  Initial Distance between C and T: {initial_distance:.2f} [km]")
		print(f"  Initial Relative velocity between C and T: {initial_velocity:.2f} [km/s]")


# defining the parameters
def get():
	param = physParamClass()
	initialValue = initialValueClass()
	initialValue.define_initialValues(param)
	initialValue.check_relative_state(param)

	return param, initialValue