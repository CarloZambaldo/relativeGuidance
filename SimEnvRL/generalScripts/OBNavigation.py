import numpy as np
import time
from .ReferenceFrames import convert_M_to_LVLH

def OBNavigation(targetState_S, chaserState_S, param):
	"""
	This function outputs the translation and rotation from Synodic to
	Moon-centered synodic and the relative state in LVLH

	"""

	# Translating and rotating to Moon-centered Synodic [ex FranziRot]
	targetState_M = np.array([
		-targetState_S[0] + (1 - param.massRatio),
		-targetState_S[1],
		targetState_S[2],
		-targetState_S[3],
		-targetState_S[4],
		targetState_S[5]
	])

	chaserState_M = np.array([
		-chaserState_S[0] + (1 - param.massRatio),
		-chaserState_S[1],
		chaserState_S[2],
		-chaserState_S[3],
		-chaserState_S[4],
		chaserState_S[5]
	])

	# Computing relative state in Moon-centered Synodic and rotating to LVLH
	relativeState_M = chaserState_M - targetState_M
	relativeState_L, _ = convert_M_to_LVLH(targetState_M, relativeState_M, param)


	## ADD DISTURBANCES HERE #
	# # Deviazione standard del rumore
	# SD = np.array([10, 0.1, 0.01])
	# 
	# # Generazione del rumore gaussiano
	# noise = np.random.normal(loc=0, scale=SD)
	#
	# targetState_M   = targetState_M + noise
	# chaserState_M   = chaserState_M + noise 
	# relativeState_L = relativeState_L + noise_relative


	return targetState_M, chaserState_M, relativeState_L
