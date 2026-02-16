clear
close all
clc

%%
% cd("C:\Users\carlo\OneDrive - Politecnico di Milano\UNIVERSITY\PROJECTS\TESI\1-CODE\simulationFull\matlabScripts\");
cd("X:\university-projects");
percorso = "/university-projects";
fprintf("Available simulations:\n")
% percorso = "../Simulations/paper/";
addpath(percorso)
simulations = ls(percorso)

%%

addpath(genpath("MATLAB"))
load("MC_P1_aposelene__Agent_P1-v11-thesis_2026_02_04_at_21_16.mat")
% % MonteCarloPlots(extractSimulationData(data,[1]),1)
MonteCarloPlots(data,1)

% % MonteCarloPlots(extractSimulationData(data,[25 26]),1)