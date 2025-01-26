clear
close all
clc
clear global OBoptimalTrajectory
global trigger OBoptimalTrajectory 


%% tune here %%
phaseID = 1;
seed = 1643734973;
tspan = [0 0.02];
trigger = 1;

%% function [timeHistory, fullStateHistory, controlActionHistory_L] = fullSimulation(phaseID)
[param,initialStateTarget_S,initialStateChaser_S,DeltaIC_S] = initializeSimulation(phaseID,tspan);
fullInitialState = [initialStateTarget_S; initialStateChaser_S];


%% RUN SIMULATION
[timeHistory,fullStateHistory,controlActionHistory_L,info] = fullSimulationFunction(fullInitialState,param);


%%
% trim the unused solution history
indicezeri = fullStateHistory(:,7:12) == 0;
fullStateHistory(indicezeri,1:6) = 0;

indiceValori = ~(fullStateHistory(:,1) == 0);
fullStateHistory = fullStateHistory(indiceValori,:);
controlActionHistory_L = controlActionHistory_L(indiceValori,:);
timeHistory = timeHistory(indiceValori);

%%
adimensionalSolution.x = timeHistory;
adimensionalSolution.y = fullStateHistory';
%adimensionalSolution.controlAction = controlActionHistory_L(1:end-1,:)';
adimensionalSolution.controlAction = controlActionHistory_L(1:end,:)';

plotty(adimensionalSolution,phaseID,param)
printSummary



