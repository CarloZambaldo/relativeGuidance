import numpy as np
from generalScripts.ReferenceFrames import convert_M_to_LVLH

def OBNavigation(targetState_S, chaserState_S, param):
	"""
	This function outputs the translation and rotation from Synodic to
	Moon-centered synodic.

	"""
	# Physical values
	rM = np.array([1 - param.massRatio, 0, 0])  # Position of the moon in Synodic frame

	# Translating and rotating to Moon-centered Synodic [ex FranziRot]
	targetState_M = np.array([
		-targetState_S[0] + rM[0],
		-targetState_S[1] + rM[1],
		targetState_S[2] - rM[2],
		-targetState_S[3],
		-targetState_S[4],
		targetState_S[5]
	])

	chaserState_M = np.array([
		-chaserState_S[0] + rM[0],
		-chaserState_S[1] + rM[1],
		chaserState_S[2] - rM[2],
		-chaserState_S[3],
		-chaserState_S[4],
		chaserState_S[5]
	])

	# Computing relative state in Moon-centered Synodic and rotating to LVLH
	relativeState_M = chaserState_M - targetState_M
	relativeState_L, _ = convert_M_to_LVLH(targetState_M, relativeState_M, param)

	return targetState_M, chaserState_M, relativeState_L
