from imports import *

global trigger
global OBoptimalTrajectory
param, initialValue = config.env_config.get()

initial_state_target_M, initial_state_chaser_M, initial_relative_state_L = generalScripts.OBNavigation(initialValue.targetState_S, initialValue.chaserState_S, param)