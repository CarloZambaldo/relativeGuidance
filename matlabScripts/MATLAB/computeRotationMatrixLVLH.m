function [R,Rdot] = computeRotationMatrixLVLH(targetState_M,param)

    % computing new reference frame axis (LVLH)
    [eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot] = versorsLVLH(targetState_M,param);

    % Rotational matrices
    R = [eR_x(:).';eV_y(:).';eH_z(:).'];
    Rdot = [eR_x_dot(:).';eV_y_dot(:).';eH_z_dot(:).'];
    %Rddot = [eR_x_ddot(:).'; eV_y_ddot(:).'; eH_z_ddot(:).'];
end