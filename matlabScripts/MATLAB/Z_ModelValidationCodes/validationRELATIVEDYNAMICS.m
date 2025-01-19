%% VALIDATION OF RELATIVE DYNAMICS
% plotting the comparison between CR3BP relative dynamics and the relative dynamics model
figure()
subplot(2,2,1);
plot(adimensionalSolution.x.*tc/60,vecnorm(relativePosition,2,2).*xc,'b-','LineWidth',1);
hold on
plot(adimensionalSolution.x.*tc/60,vecnorm(chaserPosition-targetPosition,2,2).*xc,'m--','LineWidth',1);
title("Norm of the Distance between Target and Chaser ||x||")
xlabel("time [min]");
ylabel("distance [km]");
legend("from relative dynamics","from ||r_C - r_T||_S",'location','best')
grid on

subplot(2,2,3);
semilogy(adimensionalSolution.x.*tc/60,abs(vecnorm(relativePosition,2,2)-vecnorm(chaserPosition-targetPosition,2,2)).*xc,'LineWidth',1);
hold on
%semilogy([adimensionalSolution.x(1).*tc/60, adimensionalSolution.x(end).*tc/60],[0 0],'r--','LineWidth',1);
title("Absolute error: relative vs synodic propagation abs(||x||-||x_{synodic}||)")
xlabel("time [min]");
ylabel("error [km]");
grid on

% CHECK AD UN PUNTO INTERMEDIO
error = zeros(length(adimensionalSolution.x),6);
for id = 1:length(adimensionalSolution.x)
    synodicRelState = sol(id,7:12)-sol(id,1:6);
    [rotatedRelativeState] = rotate_S_to_LVLH(sol(id,1:6)',synodicRelState',param);
    error(id,1:6) = abs(rotatedRelativeState(:)'-sol(id,13:18));
end
subplot(2,2,[2,4])
semilogy(adimensionalSolution.x.*tc/60,[error(:,1:3)*xc,error(:,4:6)*xc/tc],'LineWidth',1);
legend("R","V","H","v_R","v_V","v_H",'location','best')
title("absolute error in the computation - FOR DEBUG -")
xlabel("time [min]")
ylabel("Error [km] or [km/h]")
grid on

%% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
figure
subplot(1,2,1)
plot3(relativePosition(:,1)*xc,relativePosition(:,2)*xc,relativePosition(:,3)*xc,'b','LineWidth',1.1)
grid on
hold on
plot3(relativePosition(1,1)*xc,relativePosition(1,2)*xc,relativePosition(1,3)*xc,'ok','LineWidth',1)
plot3(0,0,0,'r*','LineWidth',2)

DeltaIC_L = rotate_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
DeltaICm = DeltaIC_L(1:3)*xc;
quiver3(0,0,0,DeltaICm(1),DeltaICm(2),DeltaICm(3),'k','LineWidth',.9)

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

legend("Relative Dynamics","Initial Condition","Target Position",'Location','best')
axis equal
title("Relative Dynamics in Target LVLH")
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")



% rotate the nonlinear equations described in SYNODIC frame
fullInitialState = [initialStateTarget_S; initialStateChaser_S; recall];
relasol = ode113(@(t,state)DEBUGactualRelativeDynamics(t,state,param),tspan,fullInitialState,odeopts);  % <<<

relDynami = zeros(length(relasol.x),6);
soluz = relasol.y';
for id = 1:length(relasol.x) 
    [rotatedRelativeState] = rotate_S_to_LVLH(soluz(id,1:6)',soluz(id,13:18)',param);
    relDynami(id,1:6) = rotatedRelativeState;
end

relDynami = relDynami.*xc;
subplot(1,2,2)
plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2);
hold on
plot3(0,0,0,'r*','LineWidth',2)
DeltaIC_L = rotate_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
DeltaICm = DeltaIC_L(1:3)*xc;
quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)

quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)

legend("Relative Dynamics","","Initial Condition","","Target LVLH",'Location','best')
axis equal
xlabel("R-bar [km]")
ylabel("V-bar [km]")
zlabel("H-bar [km]")

title("Actual Relative Dynamics - FOR DEBUG - [NO CONTROL]")
grid on



%%  plotting the relative dynamics x y z
figure()
subplot(3,1,1);
plot(adimensionalSolution.x.*tc/60/60,relativePosition(:,1).*xc,'b-','LineWidth',1);
hold on
plot(relasol.x.*tc/60/60,relDynami(:,1),'m--','LineWidth',1);
title("Relative R-BAR position")
xlabel("time [hours]");
ylabel("distance [km]");
grid on

subplot(3,1,2);
plot(adimensionalSolution.x.*tc/60/60,relativePosition(:,2).*xc,'b-','LineWidth',1);
hold on
plot(relasol.x.*tc/60/60,relDynami(:,2),'m--','LineWidth',1);
title("Relative V-BAR position")
xlabel("time [hours]");
ylabel("distance [km]");
grid on

subplot(3,1,3);
plot(adimensionalSolution.x.*tc/60/60,relativePosition(:,3).*xc,'b-','LineWidth',1);
hold on
plot(relasol.x.*tc/60/60,relDynami(:,3),'m--','LineWidth',1);
title("Relative H-BAR position")
xlabel("time [hours]");
ylabel("distance [km]");
grid on

%% integrate the linearized equations

[t,LNLRE_S] = ode113(@(t,state)relativeDynamicsLNLRESynodicPLUSTARGETDYNAMICS(t,state(1:6),state(7:12),param),adimensionalSolution.x,[initialRelativeState_S,initialStateTarget_S],odeopts);

LNLRE_S = LNLRE_S(:,1:6);

figure()
subplot(2,2,1);
plot(adimensionalSolution.x.*tc/60,vecnorm(relativePosition,2,2).*xc,'b-','LineWidth',1);
hold on
plot(adimensionalSolution.x.*tc/60,vecnorm(LNLRE_S(:,1:3),2,2).*xc,'m--','LineWidth',1);
title("Norm of the Distance between Target and Chaser ||x||")
xlabel("time [min]");
ylabel("distance [km]");
legend("from relative dynamics","from linearized synodic NLRE",'location','best')
grid on

subplot(2,2,3);
semilogy(adimensionalSolution.x.*tc/60,abs(vecnorm(relativePosition,2,2)-vecnorm(LNLRE_S(:,1:3),2,2)).*xc,'LineWidth',1);
hold on
%semilogy([adimensionalSolution.x(1).*tc/60, adimensionalSolution.x(end).*tc/60],[0 0],'r--','LineWidth',1);
title("Absolute error: relative vs LNLRE synodic propagation abs(||x||-||x_{synodic}||)")
xlabel("time [min]");
ylabel("error [km]");
grid on

% CHECK AD UN PUNTO INTERMEDIO
error = zeros(length(adimensionalSolution.x),6);
for id = 1:length(adimensionalSolution.x)
    synodicRelState = LNLRE_S(id,1:6);
    [rotatedRelativeState] = rotate_S_to_LVLH(sol(id,1:6)',synodicRelState',param);
    error(id,1:6) = abs(rotatedRelativeState(:)'-sol(id,13:18));
end
subplot(2,2,[2,4])
semilogy(adimensionalSolution.x.*tc/60,[error(:,1:3)*xc,error(:,4:6)*xc/tc],'LineWidth',1);
legend("R","V","H","v_R","v_V","v_H",'location','best')
title("absolute error in the computation - FOR DEBUG -")
xlabel("time [min]")
ylabel("Error [km] or [km/h]")
grid on