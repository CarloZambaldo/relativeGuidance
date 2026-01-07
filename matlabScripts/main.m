clear
close all
clc

%%
fprintf("Available simulations:\n")
percorso = "../Simulations/paper/";
addpath(percorso)
simulations = ls(percorso)

%%

addpath(genpath("MATLAB"))
load("MC_P1_approaching_aposelene__Agent_P1_v5_2025_12_18_at_11_53.mat")
% % MonteCarloPlots(extractSimulationData(data,[1]),1)
MonteCarloPlots(data,1)

% % MonteCarloPlots(extractSimulationData(data,[25 26]),1)