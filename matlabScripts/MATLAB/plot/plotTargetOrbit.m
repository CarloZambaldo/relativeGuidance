clear
close all
clc


%% initial conditions
tspan = [0 4*pi/9]; % full orbit
[param,~,~,~] = initializeSimulation(1,tspan,0);
initialStateTarget_S = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]';

%% integrate orbit
odeopts = odeset("AbsTol",1e-11,"RelTol",1e-10);                                                    
[adimensionalSolution] = ode113(@(t,state)CR3BP(t,state,param),tspan,initialStateTarget_S,odeopts); 

%% plot orbit
figure
plotDimensional(adimensionalSolution,param); 