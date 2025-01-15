import pickle
from SimEnvRL import *
import matlab.engine

fileNumber = "1736928017"

with open(f"Simulations/{fileNumber}.pkl", "rb") as file:
    env = pickle.load(file)
    
printSummary(env.unwrapped)
plotty(env.unwrapped)


## MATLAB ##
# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Run the MATLAB script
print("Plotting ...")
eng.matlabPLOT(env.unwrapped.getHistory(), 1, nargout=0)
print("Done. Press any key to close the plots ...")
# Stop MATLAB engine
eng.quit()

# Save the simulation environment to a .mat file
# scipy.io.savemat("Simulations/env.mat", {"env": env.unwrapped.saveToMAT()})