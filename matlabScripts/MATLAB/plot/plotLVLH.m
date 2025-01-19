function [] = plotLVLH(timeID, adimensionalSolution,param)

    solution.y = adimensionalSolution.y';
    stateVar = solution.y(timeID,1:6);
    
    rTM = stateVar(1:3)' - [1-param.massRatio; 0; 0];
    vTM = stateVar(4:6)' - cross([0;0;1],[1-param.massRatio; 0; 0]);

    [eR_x, eV_y, eH_z] = versorsLVLH(targetState,param);

    hold on
    plot3(stateVar(1),stateVar(2),stateVar(3),'r.','LineWidth',500)

    quiver3(stateVar(1),stateVar(2),stateVar(3),eR_x(1)*0.1,eR_x(2)*0.1,eR_x(3)*0.1,'k','LineWidth',1);
    quiver3(stateVar(1),stateVar(2),stateVar(3),eV_y(1)*0.1,eV_y(2)*0.1,eV_y(3)*0.1,'k','LineWidth',1);
    quiver3(stateVar(1),stateVar(2),stateVar(3),eH_z(1)*0.1,eH_z(2)*0.1,eH_z(3)*0.1,'k','LineWidth',1);
end