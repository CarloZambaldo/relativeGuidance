## Configuration for the Reinforcement Learning ##

from dataclasses import dataclass, field
import numpy as np
import scipy.io
import random




@dataclass()
class RLParamClass():

	lea : int = 10000




	
# defining the parameters
def get():
	RLParam = RLParamClass()

	return RLParam