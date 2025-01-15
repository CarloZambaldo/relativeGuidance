## Configuration for the Reinforcement Learning ##
from dataclasses import dataclass, field
import os
import time
import multiprocessing
import subprocess


# Define the function outside the class for multiprocessing compatibility
def run_tensorboard(logdir):
    """Runs TensorBoard in a subprocess."""
    subprocess.Popen(['tensorboard', '--logdir', logdir, '--host', 'localhost', '--port', '6006'])


@dataclass()
class RLagentParamClass():
    modelName  : str = ""
    model_dir : str = ""
    log_dir    : str = ""
    maxTimeSteps : int = 0

    def define(self, modelName):
        self.timeStamp = int(time.time())
        self.model_dir = f"AgentModels/{modelName}/model/{self.timeStamp}.zip"
        self.log_dir    = f"AgentModels/{modelName}/logs/{self.timeStamp}"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.maxTimeSteps = 110000 # about 3 hours of environmet time
        
        return self

    def latest(self, modelName):
        model_dir = f"AgentModels/{modelName}/model"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No models found in directory: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {model_dir}")
        
        latest_model =  max([int((f.split('.')[0]).strip('{}')) for f in model_files if f.endswith('.zip')])

        self.model_dir = f"{model_dir}/{latest_model}.zip"
        self.log_dir = f"AgentModels/{modelName}/logs/{latest_model}"

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        return self

    def open(self, modelName, modelNumber):
        model_dir = f"AgentModels/{modelName}/model/"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No models found in directory: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {model_dir}")

        self.model_dir = f"AgentModels/{modelName}/model/{modelNumber}.zip"
        self.log_dir = f"AgentModels/{modelName}/logs/{modelNumber}/"
        
        return self


    def viewLogs(self):
        ##"""Launches TensorBoard logs in a parallel process."""
        ##training_log_dir = os.path.join(self.log_dir)
        ### Use the external function for multiprocessing compatibility
        ##p = multiprocessing.Process(target=run_tensorboard, args=(training_log_dir,))
        ##p.start()


        ## USE INSTEAD: tensorboard --logdir="AgentModels//" --host localhost --port 6006
        pass

# defining the parameters
def get(modelName):
    RLagent = RLagentParamClass()
    RLagent = RLagent.define(modelName)
    return RLagent

# recalling previous model
def recall(modelName,modelNumber):
    RLagent = RLagentParamClass()
    if modelNumber.upper() == 'LATEST':
        RLagent = RLagent.latest(modelName)
    else:
        RLagent = RLagent.open(modelName,modelNumber)

    return RLagent
