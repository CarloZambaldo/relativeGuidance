from SimEnvRL import *
from stable_baselines3 import PPO
from datetime import datetime
import cProfile
import pstats
from line_profiler import LineProfiler

def function_to_profile():
    ## PARAMETERS THAT CAN BE CHANGED:
    phaseID = 2
    tspan = np.array([0, 0.02])
    agentName = "Agent_P2-PPO-v5-CUDA"
    renderingBool = True

    # make the environment
    env = gym.make("SimEnv-v2", options={"phaseID":phaseID,"tspan":tspan})

    # load the model
    RLagent = config.RL_config.recall(agentName,"latest")
    model = PPO.load(f"{RLagent.model_dir}\\{RLagent.modelNumber}", env=env, device="cpu")

    terminated = False
    truncated = False
    obs, info = env.reset()
    
    while ((not terminated) and (not truncated)):
        action, _ = model.predict(obs) # predict the action using the agent
        obs, reward, terminated, truncated, info = env.step(1) # step
        if renderingBool:
            print(env.render())
        truncated = True



## GENERAL PROFILING
profiler = cProfile.Profile()
profiler.run('function_to_profile()')
with open('profiling_results.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('time').print_stats()
with open('profiling_results_short.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('cumtime').print_stats(10)

## LINE PROFILING
profiler = LineProfiler()
profiler.add_function(function_to_profile())
profiler.run('function_to_profile()')
with open('profiling_results_lineprofiler.txt', 'w') as f:
     profiler.print_stats(stream=f)

