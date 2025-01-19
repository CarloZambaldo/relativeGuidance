function [rotatedState_S] = convert_LVLH_to_S(targetState_S,stateToBeRotated_L,param)
    % to rotate from S to LVLH first a translation is needed, then it is
    % possible to rotate from S to M and eventually rotate from M to LVLH

    % translate from Synodic to Moon and rotating from Moon to Moon Synodic [T14]
    rM = [1-param.massRatio;0;0]; % position of the moon in Synodic frame
    targetState_SCM = targetState_S-[rM(:);0;0;0];
    FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
    targetState_M = FranziRot*targetState_SCM;

    %% rotate from LVLH to M

    % compute LVLH versors from Moon centered synodic
    [eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot] = versorsLVLH(targetState_M,param);

    % rotating matrices
    R = [eR_x(:).';eV_y(:).';eH_z(:).'];
    Rdot = [eR_x_dot(:).';eV_y_dot(:).';eH_z_dot(:).'];
    Rtot = [R zeros(3); Rdot, R];

    % rotating frame
    rotatedState_SCM = (Rtot\eye(6))*stateToBeRotated_L;

    FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
    rotatedState_S = FranziRot*rotatedState_SCM;

end