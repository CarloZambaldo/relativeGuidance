function [rotatedState] = convert_S_to_LVLH(targetState_S,stateToBeRotated_S,param)
    % to rotate from S to LVLH first a translation is needed, then it is
    % possible to rotate from S to M and eventually rotate from M to LVLH

    % translate from Synodic to Moon
    rM = [1-param.massRatio;0;0]; % position of the moon in Synodic frame
    targetState_M = targetState_S-[rM(:);0;0;0];

    % rotating from Moon to Moon Synodic [T14]
    FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
    targetState_M = FranziRot*targetState_M;
    stateToBeRotated_M = FranziRot*stateToBeRotated_S;

    % rotating frame
    [rotatedState] = convert_M_to_LVLH(targetState_M,stateToBeRotated_M,param);
end