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
load("MC_P2_periselene___NO_AGENT__2026_01_27_at_16_59.mat")
% MonteCarloPlots(extractSimulationData(data,[26]),2)
MonteCarloPlots(data,2)
% MonteCarloInfo(data);
% % MonteCarloPlots(extractSimulationData(data,[25 26]),1)