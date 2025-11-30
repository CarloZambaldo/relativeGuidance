fprintf("Available simulations:\n")
addpath("../Simulations/")
simulations = ls("../Simulations")

addpath(genpath("MATLAB"))
load("MC_P2__Agent_P2-v11.5-multi-SEMIDEF_2025_11_25_at_13_11.mat")
% % MonteCarloPlots(extractSimulationData(data,[1]),1)
MonteCarloPlots(data,1)