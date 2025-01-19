param.xc = 384400;
param.tc = 1/(2.661699e-6); % (2*pi/2358720);

targetState_S = [ 1.01056035, -0.03617079, -0.13739958, -0.05186586, -0.05896856, 0.22699674]';
chaserState_S = [ 1.01054142, -0.03616771, -0.13740013, -0.0486426 , -0.05719729, 0.22417574]';

rM = [1-param.massRatio;0;0]; % position of the moon in Synodic frame
targetState_M = targetState_S-[rM(:);0;0;0];

[targetState_M,chaserState_M,relativeState_L] = OBNavigation(targetState_S,chaserState_S,param)


TOF = norm(relativeState_L(1:3))/(1e-3/param.xc*param.tc)*1.1
%%
optimalTrajectory = ASRE(TOF, initialRelativeState_L, initialStateTarget_M, finalAimState, phaseID, param)
