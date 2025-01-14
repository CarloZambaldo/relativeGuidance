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

    def define(self, modelName):
        self.timeStamp = int(time.time())
        self.models_dir = f"AgentModels/{modelName}/model/{self.timeStamp}/"
        self.log_dir    = f"AgentModels/{modelName}/logs/{self.timeStamp}/"
        #if not os.path.exists(self.models_dir):
        #    os.makedirs(self.models_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.maxTimeSteps = 110000 # about 3 hours of environmet time
        
        return self

    def latest(self, modelName):
        model_dir = f"AgentModels/{modelName}/model/"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No models found in directory: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {model_dir}")
        
        latest_model =  max([int((f.split('.')[0]).strip('{}')) for f in model_files if f.endswith('.zip')])

        self.models_dir = f"{model_dir}/{latest_model}"
        self.log_dir = f"AgentModels/{modelName}/logs/{latest_model}/"

        os.makedirs(os.path.dirname(self.models_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        return self

    def open(self, modelName, modelNumber):
        model_dir = f"AgentModels/{modelName}/model/"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No models found in directory: {model_dir}")
        
        model_files = [f for f in os.listdir(model_dir)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in directory: {model_dir}")

        self.models_dir = f"AgentModels/{modelName}/model/{modelNumber}.zip/"
        self.log_dir = f"AgentModels/{modelName}/logs/{modelNumber}/"
        
        return self


    def viewLogs(self):
        ## VIEW THE TENSORBOARD LOGS ##
        training_log_dir = os.path.join(self.log_dir)
        import subprocess
        import multiprocessing
        def run_tensorboard(logdir):
            subprocess.Popen(['tensorboard', '--logdir', logdir, '--host', 'localhost', '--port', '6006'])
        p = multiprocessing.Process(target=run_tensorboard, args=(training_log_dir,))
        p.start()

# defining the parameters
def get(modelName):
    RLParam = RLParamClass()
    RLParam = RLParam.define(modelName)
    return RLParam

# recalling previous model
def recall(modelName,modelNumber):
    RLParam = RLParamClass()
    if modelNumber.upper() == 'LATEST':
        RLParam = RLParam.latest(modelName)
    else:
        RLParam = RLParam.open(modelName,modelNumber)
        
    return RLParam
