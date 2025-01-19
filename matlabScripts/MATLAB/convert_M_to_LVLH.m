function [rotatedState,Rtot] = convert_M_to_LVLH(targetState_M,stateToBeRotated,param)
    % compute LVLH versors from Moon centered synodic
    [eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot] = versorsLVLH(targetState_M,param);

    % rotating matrices
    R = [eR_x(:).';eV_y(:).';eH_z(:).'];
    Rdot = [eR_x_dot(:).';eV_y_dot(:).';eH_z_dot(:).'];
    Rtot = [R zeros(3); Rdot, R];

    % rotating frame
    rotatedState = Rtot*stateToBeRotated;
end