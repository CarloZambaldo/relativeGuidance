clear
close all
clc

%%
cd("C:\Users\carlo\OneDrive - Politecnico di Milano\UNIVERSITY\PROJECTS\TESI\1-CODE\simulationFull\matlabScripts\");
fprintf("Available simulations:\n")
percorso = "../Simulations/paper/";
addpath(percorso)
simulations = ls(percorso)

%%

addpath(genpath("MATLAB"))
load("MC_P2_aposelene__Agent_P2-v11.5-multi-SEMIDEF_2026_01_28_at_10_10.mat")
% MonteCarloPlots(extractSimulationData(data,[26]),2)
MonteCarloPlots(data,2)
% MonteCarloInfo(data);
% % MonteCarloPlots(extractSimulationData(data,[25 26]),1)