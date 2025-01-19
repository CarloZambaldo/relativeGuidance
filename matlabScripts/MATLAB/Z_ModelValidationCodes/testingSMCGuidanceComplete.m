clear
close all
clc
clear global OBoptimalTrajectory
global trigger OBoptimalTrajectory 

%% tune here %%
phaseID = 2;
tspan = [0 .02];
trigger = 1;

%%
[param,initialStateTarget_S,initialStateChaser_S,DeltaIC_S] = initializeSimulation(phaseID);

%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
fullInitialState = [initialStateTarget_S; initialStateChaser_S];

odeopts = odeset("AbsTol",1e-9,"RelTol",1e-9,"Events",@(t,state)checkDocking(t,state,param));                                                        % <<<
[time, states] = ode78(@(t,state)guidanceSimEquations(t,state,phaseID,param),tspan,fullInitialState,odeopts);  % <<<

adimensionalSolution.x = time';
adimensionalSolution.y = states';

trigger = 1;
[~,U] = cellfun(@(t,state)guidanceSimEquations(t,state,phaseID,param), num2cell(adimensionalSolution.x), num2cell(adimensionalSolution.y,1),'uni',0);
adimensionalSolution.controlAction  = cell2mat(U);
fprintf("Integration Finished.\n\n")
figure
plotDimensional(adimensionalSolution,param);                                                          % <<<
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


%% PLOTS & summary
plotty(adimensionalSolution,phaseID,param)

printSummary