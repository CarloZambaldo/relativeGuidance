clear
close all
clc

[param,targetState_S,~,~] = initializeSimulation(1);

testState_L = [1 2 3 4 5 6]'
[prova_S] = convert_LVLH_to_S(targetState_S,testState_L,param)
[final_L] = convert_S_to_LVLH(targetState_S,prova_S,param)


return

%%%%%%%%
targetState_M = targetState_S - [1-param.massRatio;0;0;0;0;0];
FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
targetState_M = FranziRot*targetState_M;

timeHistory = [0:.001:1];

[~,odesol1] = ode113(@(t,state)CR3BP(t, state, param),timeHistory,targetState_S);
[~,odesol2] = ode113(@(t,state)CR3BP_MoonFrame(t, state, param),timeHistory,targetState_M);

for indx = 1:size(odesol2,1)
    FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
    odesol2(indx,:) = (FranziRot*odesol2(indx,:)')';
    odesol2(indx,:) = odesol2(indx,:)+[1-param.massRatio;0;0;0;0;0]';
end

figure
subplot(3,1,1)
plot(odesol1(:,1:3))
grid on
subplot(3,1,2)
plot(odesol2(:,1:3))
grid on
subplot(3,1,3)
plot(odesol2(:,:)-odesol1(:,:))
grid on
