from imports import *

class AgentActions(Enum):
	OPTILOOP_COMP = 1
	OPTILOOP_SKIP = 0

## REINFORCEMENT LEARNING ENVIRONMENT ##
class RLenv(gym.Env):
	# DO NOT FORGET TO ADD METADATA:
	metadata = {"render_modes": ["none"]}

	def __init__(self):
		super(RLenv,self).__init__()

		## OBSERVATION SPACE
		# observations shall provide information about the location of the agent and target 
		# self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
		# Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
		self.observation_space = spaces.Dict(
			{
				"chaser": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
				"target": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
				"relative": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
			}
		)
		self.chaserState = np.array([-np.inf, -1], dtype=np.float64)
		self.targetState = np.array([-np.inf, -1], dtype=np.float64)

		## ACTION SPACE
		# 2 actions are present : OPTILOOP_COMP and OPTILOOP_SKIP 
		self.action_space = spaces.Discrete(2)

	def step(self, action):


		if collision_check(self.chaserState, self.targetState):
			self.done = True
			self.reward = -1

			
		return self.observation, self.reward, self.done, self.info
	

	def _get_obs(self):
		return {"chaser": self.chaserState, "target": self.targetState}


	def _get_info(self):
		return {
			"relativeState": self.chaserState - self.targetState
		}

	# The reset method should set the initial state of the environment (e.g., relative position and velocity) and return the initial observation.
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.done = False
		# Set the initial state of the pysical environment
		self.param, self.initialValue = config.env_config.get()

		
		return self.observation
