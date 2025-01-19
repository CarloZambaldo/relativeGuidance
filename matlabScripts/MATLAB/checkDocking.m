function [value,isterminal,direction] = checkDocking(t,state,param)
    direction = 0;
    [~,~,relativeState_L] = OBNavigation(state(1:6),state(7:12),param);

    value = 1;
    if (norm(relativeState_L(1:3)-param.dockingState(1:3)) < 3e-9) && (norm(relativeState_L(4:6)-param.dockingState(4:6)) < 6e-5)
        value = 0;
        fprintf("\n >> DOCKING SUCCESSFUL << \n");
    end
    isterminal = 1;
end