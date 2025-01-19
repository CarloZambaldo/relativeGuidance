function [eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot] = versorsLVLH(targetState_M,param)
    %
    % [eR_x, eV_y, eH_z, eR_x_dot, eV_y_dot, eH_z_dot] = versorsLVLH(targetState_M,param)
    % 
    % this function computes the LVLH versors in the Moon centered frame
    % and its time derivative, this function can be used to build the
    % rotation matrices to rotate from Moon centered Synodic to LVLH and
    % vice versa
    %
    % 

    % from MOON CENTERED synodic values
    rTM = targetState_M(1:3,:);
    vTM = targetState_M(4:6,:);
    dSt = CR3BP_MoonFrame(0,[targetState_M(1:6,1)],param);
    aTM = dSt(4:6);

    % other values
    hTM = cross(rTM,vTM);
    h = norm(cross(rTM,vTM));
    hdot = dot(cross(rTM,aTM),cross(rTM,vTM))/h;

    % computing new reference frame axis (LVLH)
    eR_x = rTM/norm(rTM);
    eH_z = cross(rTM,vTM)/norm(cross(rTM,vTM));
    eV_y = cross(eH_z,eR_x);
    
    % derivatives of the reference frame axis (LVLH)
    eR_x_dot = 1/norm(rTM)*(vTM'*eV_y)*eV_y;
    eH_z_dot = -norm(rTM)/norm(hTM)*(aTM'*eH_z)*eV_y; % before it was:  (h*cross(rTM,aTM) - hdot*cross(rTM,vTM))/h^2
    eV_y_dot = cross(eH_z_dot,eR_x) + cross(eH_z,eR_x_dot);  % before it was: norm(rTM)/norm(hTM)*(aTM'*eH_z)*eH_z - 1/norm(rTM)*(vTM'*eV_y)*eR_x;
end