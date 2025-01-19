clear
clc
close all


fprintf("COMPUTING REFERENCE TRAJECTORY PLEASE WAIT....\n")
tic
%% ADIMENSIONALISERS
param.xc = 384400;
param.tc = 1/(2.661699e-6); % (2*pi/2358720);


%% Simulation Parameters
% This is the delta_state between C and T (applied to T only) at the beginning of the simulation
initialStateTarget_S = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]'; % see PhD thesis

% Astronomical Parameters
earth = ASTRO("EARTH");
moon  = ASTRO("MOON");

massEarth = earth.mass;
massMoon = moon.mass;
param.massRatio = massMoon/(massEarth+massMoon);
param.Omega = 2*pi/2358720;
param.guidanceBool = 0;

earth = earth.updatePosition([param.massRatio, 0, 0]);
moon = moon.updatePosition([1-param.massRatio, 0, 0]);

param.moon = moon;
param.earth = earth;

tspan = linspace(0, 4*pi/9,1000001);
refTraj.AbsTol = 1e-11;
refTraj.RelTol = 1e-10;

odeopts = odeset("AbsTol",refTraj.AbsTol,"RelTol",refTraj.RelTol);         
[t,y] = ode113(@(t,state)CR3BP(t,state,param),tspan,initialStateTarget_S,odeopts);

refTraj.y = y';
refTraj.x = t';
refTraj.solver = 'ode113';

save("refTraj.mat","refTraj");

fprintf("REFERENCE TRAJECTORY COMPUTED. [Elapsed time: %.2f sec]\n",toc)