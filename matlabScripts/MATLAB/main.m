clear
close all
clc

%% initialize Simulation in Synodic Frame
initializeSimulation;
xc = param.xc;
tc = param.tc;

recall = initialRelativeState_S;
% [targetState_M,chaserState_M] = OBNavigation(initialStateTarget_S,initialStateChaser_S,param);
[initialRelativeState_L] = convert_S_to_LVLH(initialStateTarget_S,initialRelativeState_S,param);


%% INTEGRATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
fullInitialState = [initialStateTarget_S; initialStateChaser_S; initialRelativeState_L];
tspan = [0 .6];%[0 4*pi/9];
param.BOOLS.controlActionBool = 1;

odeopts = odeset("AbsTol",1e-11,"RelTol",1e-10);                                                        % <<<
[adimensionalSolution] = ode113(@(t,state)simEquations(t,state,param),tspan,fullInitialState,odeopts);  % <<<
[~,U] = cellfun(@(t,state)simEquations(t,state,param), num2cell(adimensionalSolution.x), num2cell(adimensionalSolution.y,1),'uni',0);
figure
plotDimensional(adimensionalSolution,param);                                                          % <<<
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%% VALUES
sol = adimensionalSolution.y';
chaserPosition   = sol(:,7:9);
targetPosition   = sol(:,1:3);
relativePosition = sol(:,13:15);

%% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
% figure
% plot3(relativePosition(:,1)*xc,relativePosition(:,2)*xc,relativePosition(:,3)*xc,'b','LineWidth',1.1)
% grid on
% hold on
% plot3(relativePosition(1,1)*xc,relativePosition(1,2)*xc,relativePosition(1,3)*xc,'ok','LineWidth',1)
% plot3(0,0,0,'r*','LineWidth',2)
% 
% DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
% DeltaICm = DeltaIC_L(1:3)*xc;
% quiver3(0,0,0,DeltaICm(1),DeltaICm(2),DeltaICm(3),'k','LineWidth',.9)
% 
% resaizer = max(relativePosition)*xc/2;
% quiver3(0,0,0,norm(resaizer),0,0,'r','LineWidth',1)
% quiver3(0,0,0,0,norm(resaizer),0,'r','LineWidth',1)
% quiver3(0,0,0,0,0,norm(resaizer),'r','LineWidth',1)
% 
% plotConstraintsVisualization(DeltaIC_S)
% 
% legend("Relative Dynamics","Initial Condition","Target Position",'Location','best')
% axis equal
% title("Relative Dynamics in Target LVLH [PROPAGATED IN LVLH]")
% xlabel("R-bar [km]")
% ylabel("V-bar [km]")
% zlabel("H-bar [km]")

%% %% %% %%
relDynami = zeros(length(adimensionalSolution.x),6);
soluz = adimensionalSolution.y';
for id = 1:length(adimensionalSolution.x) 
    [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
    relDynami(id,1:6) = rotatedRelativeState;
end

relDynami = relDynami.*xc;
figure
plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2);
hold on
plot3(0,0,0,'r*','LineWidth',2)
DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
DeltaICm = DeltaIC_L(1:3)*xc;
quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

plotConstraintsVisualization(DeltaIC_S./10)

legend("Relative Dynamics","","Initial Condition","","Target LVLH",'Location','best')
axis equal
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")

title("Actual Relative Dynamics [CONTROLLED]")
grid on

%%
figure
subplot(3,1,1)
controlAction = cell2mat(U);
plot(adimensionalSolution.x*param.tc/60, controlAction,'LineWidth',1.1)

grid on;
title("control Action")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
xlabel("Time [min]")
ylabel("control Action [LVLH]")

subplot(3,1,2)
plot(adimensionalSolution.x*param.tc/60,relDynami(:,1),'LineWidth',1);
hold on
plot(adimensionalSolution.x*param.tc/60,relDynami(:,2),'LineWidth',1);
plot(adimensionalSolution.x*param.tc/60,relDynami(:,3),'LineWidth',1);
xlabel("Time [min]")
ylabel("position [km]")
title("Controlled Relative Dynamics [LVLH]")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
grid on

subplot(3,1,3)
plot(adimensionalSolution.x*param.tc/60,relDynami(:,4)/param.tc,'LineWidth',1);
hold on
plot(adimensionalSolution.x*param.tc/60,relDynami(:,5)/param.tc,'LineWidth',1);
plot(adimensionalSolution.x*param.tc/60,relDynami(:,6)/param.tc,'LineWidth',1);
xlabel("Time [min]")
ylabel("velocity [km/s]")
title("Controlled Relative Velocity [LVLH]")
legend("R-BAR","V-BAR","H-BAR",'Location','best')
grid on

%% VALIDATION OF RELATIVE DYNAMICS
%validationRELATIVEDYNAMICS


%% %% %% %% MAGIC
relDynami = zeros(length(adimensionalSolution.x),6);
soluz = adimensionalSolution.y';
for id = 1:length(adimensionalSolution.x) 
    [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
    relDynami(id,1:6) = rotatedRelativeState;
end

relDynami = relDynami.*xc;
figure
plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2);
hold on
plot3(0,0,0,'r*','LineWidth',2)
DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
DeltaICm = DeltaIC_L(1:3)*xc;
quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

quiver3(relDynami(:,1),relDynami(:,2),relDynami(:,3),controlAction(1,:)',controlAction(2,:)',controlAction(3,:)','g','LineWidth',0.8)
plotConstraintsVisualization(DeltaIC_S/3)

legend("Relative Dynamics","","Initial Condition","","Target LVLH",'Location','best')
axis equal
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")

title("Actual Relative Dynamics [WITH CONTROL SHOWN]")
grid on