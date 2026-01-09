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
load("MC_P1_aposelene___NO_AGENT__2026_01_07_at_15_09.mat")
% % MonteCarloPlots(extractSimulationData(data,[1]),1)
MonteCarloPlots(data,1)

% % MonteCarloPlots(extractSimulationData(data,[25 26]),1)