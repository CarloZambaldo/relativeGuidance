function [rotatedControlAction] = rotateControlAction(targetState_S,uToBeRotated_L,param)

    % TRASLATING AND ROTATING to MOON CENTERED SYNODIC
    rM = [1-param.massRatio;0;0]; % position of the moon in Synodic frame
    targetState_M = [-targetState_S(1:2)+rM(1:2); targetState_S(3)-rM(3); -targetState_S(4:5); targetState_S(6)];

    % compute LVLH versors from Moon centered synodic
    [eR_x, eV_y, eH_z] = versorsLVLH(targetState_M,param);

    % rotating matrices
    R_L_to_M = [eR_x(:).';eV_y(:).';eH_z(:).'].';

    %% rotating from L to M
    rotatedControlAction = R_L_to_M*uToBeRotated_L;
    
    %% rotating from Moon Synodic [M] to Synodic [S]
    % Rota = [-1 0 0; 0 -1 0; 0 0 1]; rotatedControlAction = (Rota')*rotatedControlAction;
    rotatedControlAction = [-rotatedControlAction(1);-rotatedControlAction(2);rotatedControlAction(3)];
end