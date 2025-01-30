clear
close all
clc

%% 
seed = 1792328962;
phaseID = 2;
[param,initialStateTarget_S,initialStateChaser_S,DeltaIC_S] = initializeSimulation(phaseID,0);
seed = param.rng_settings.Seed
%%

switch phaseID
    case 1
        aimAtState = param.holdingState;
    case 2
        aimAtState = param.dockingState;
end

[initialTargetState_M,initialStateChaser_M,initialRelativeState_L] = OBNavigation(initialStateTarget_S,initialStateChaser_S,param);

%%
TOF = computeTOF(initialRelativeState_L,aimAtState,param);

tspan = [0 TOF];
fprintf("\n  Imposed TOF: %f [-]\n",TOF);
optimalTrajectory = ASRE(TOF,initialRelativeState_L, initialTargetState_M, aimAtState, phaseID, param);


%% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
figure
plot3(optimalTrajectory.x(1,:)*param.xc,optimalTrajectory.x(2,:)*param.xc,optimalTrajectory.x(3,:)*param.xc,'b.-','LineWidth',1.1)
grid on
hold on
plot3(optimalTrajectory.x(1,1)*param.xc,optimalTrajectory.x(2,1)*param.xc,optimalTrajectory.x(3,1)*param.xc,'ok','LineWidth',1)
plot3(optimalTrajectory.x(1,end)*param.xc,optimalTrajectory.x(2,end)*param.xc,optimalTrajectory.x(3,end)*param.xc,'*k','LineWidth',1)
plot3(0,0,0,'r*','LineWidth',2)

DeltaIC_L = convert_M_to_LVLH(initialTargetState_M,initialStateChaser_M-initialTargetState_M,param);
DeltaICm = DeltaIC_L(1:3)*param.xc;

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

legend("Optimal Trajectory","Initial Chaser Position","Final Chaser Position","Target Position","LVLH",'Location','best')
axis equal
title("ONBOARD-COMPUTED Relative Dynamics in Target LVLH")
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")


%%
figure
subplot(3,1,1)
plot(optimalTrajectory.t*param.tc,optimalTrajectory.x(1,:)*param.xc,'LineWidth',1); 
grid on;
title("ONBOARD-COMPUTED Trajectory on R-bar");
subplot(3,1,2)
plot(optimalTrajectory.t*param.tc,optimalTrajectory.x(2,:)*param.xc,'LineWidth',1);
grid on;
title("ONBOARD-COMPUTED Trajectory on V-bar");
subplot(3,1,3)
plot(optimalTrajectory.t*param.tc,optimalTrajectory.x(3,:)*param.xc,'LineWidth',1);
grid on;
title("ONBOARD-COMPUTED Trajectory on H-bar");

%%
figure
subplot(3,1,1)
plot(optimalTrajectory.t*param.tc/60,optimalTrajectory.u(1,:),'LineWidth',1); 
grid on;
title("Control Action on R-bar");
xlabel("time [min]")
ylabel("Control Action [-]")
subplot(3,1,2)
plot(optimalTrajectory.t*param.tc/60,optimalTrajectory.u(2,:),'LineWidth',1);
grid on;
title("Control Action on V-bar");
xlabel("time [min]")
ylabel("Control Action [-]")
subplot(3,1,3)
plot(optimalTrajectory.t*param.tc/60,optimalTrajectory.u(3,:),'LineWidth',1);
grid on;
title("Control Action on H-bar");
xlabel("time [min]")
ylabel("Control Action [-]")
%%

relDynami = optimalTrajectory.x'*param.xc;

figure
subplot(3,1,1)
controlAction = optimalTrajectory.u';
plot(optimalTrajectory.t*param.tc/60, controlAction,'LineWidth',1.1)

grid on;
title("control Action [LVLH]")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
xlabel("Time [min]")
ylabel("control Action [-]")

subplot(3,1,2)
plot(optimalTrajectory.t*param.tc/60,relDynami(:,1),'LineWidth',1);
hold on
plot(optimalTrajectory.t*param.tc/60,relDynami(:,2),'LineWidth',1);
plot(optimalTrajectory.t*param.tc/60,relDynami(:,3),'LineWidth',1);
xlabel("Time [min]")
ylabel("position [km]")
title("Controlled Relative Dynamics [LVLH]")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
grid on

subplot(3,1,3)
plot(optimalTrajectory.t*param.tc/60,relDynami(:,4)/param.tc,'LineWidth',1);
hold on
plot(optimalTrajectory.t*param.tc/60,relDynami(:,5)/param.tc,'LineWidth',1);
plot(optimalTrajectory.t*param.tc/60,relDynami(:,6)/param.tc,'LineWidth',1);
xlabel("Time [min]")
ylabel("velocity [km/s]")
title("Controlled Relative Velocity [LVLH]")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
grid on


%%
figure
plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2);
hold on
plot3(0,0,0,'r*','LineWidth',2)
DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
DeltaICm = DeltaIC_L(1:3)*param.xc;
quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

quiver3(relDynami(:,1),relDynami(:,2),relDynami(:,3),controlAction(:,1),controlAction(:,2),controlAction(:,3),'g','LineWidth',0.8)

    
    if phaseID == 1
        plotConstraintsVisualization(1e3,'S','yellow')
        plotConstraintsVisualization(200,'S')
        plotConstraintsVisualization(2.5e3,'S','#808080')
    elseif phaseID == 2
        plotConstraintsVisualization(1e3,'C')
    end

legend("Relative Dynamics","","Initial Condition","","Target LVLH",'Location','best')
axis equal
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")

title("Actual Relative Dynamics [WITH CONTROL SHOWN]")
grid on

%% 
trajectory = optimalTrajectory;
trajectory.x = trajectory.x.*param.xc*1e3;
switch(phaseID)
    case 1 %% REACHING SAFE HOLD
        constraintType = 'SPHERE';
        aimAtState = param.holdingState;
        characteristicSize = 200;
    case 2 %% DOCKING PHASE
        constraintType = 'CONE';
        aimAtState = param.dockingState;
        characteristicSize.acone = 0.08;
        characteristicSize.bcone = 5;
    otherwise
        error("Wrong phase ID");
end
[violationFlag,violationPosition] = checkConstraintViolation(trajectory,constraintType,characteristicSize);
