## Configuration for the Reinforcement Learning ##
from dataclasses import dataclass, field
import os
import time

@dataclass()
class RLParamClass():
    modelName  : str = ""
    models_dir : str = ""
    log_dir    : str = ""
    maxTimeSteps : int = 0

    def define(self,modelName):
        self.timeStamp = {int(time.time())}
        self.models_dir = f"AgentModels/{modelName}/model/{self.timeStamp}"
        self.log_dir    = f"AgentModels/{modelName}/logs/{self.timeStamp}"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.maxTimeSteps = 110000 # about 3 hours of environmet time
        
        return self

    def viewLogs(self):
        ## VIEW THE TENSORBOARD LOGS ##
        training_log_dir = os.path.join(self.log_dir)
        os.system(f"tensorboard --logdir={training_log_dir} --host localhost --port 6006")


# defining the parameters
def get(modelName):
    RLParam = RLParamClass()
    RLParam = RLParam.define(modelName)
    return RLParam


