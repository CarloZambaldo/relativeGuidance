function [dRelState] = relativeDynamicsLVLH(t,relativeState_L,targetState_M,chaserState_M,param,controlAction_L)
    %   
    % [dRelState] = relativeDynamicsLVLH(t,relativeState_L,targetState_M,chaserState_M,param)
    %
    % this function outputs the RHS of the relative dynamics equation found
    % in Franzini work

    %% EXTRACTING VALUES FROM INPUT
    % environment data [in Moon Synodic]
    massRatio = param.massRatio;
    omegaMI = [0;0;1];
    rem = [-1;0;0];

    % chaser data
    rCM = chaserState_M(1:3);
    rCE = rCM + rem;
    vCM = chaserState_M(4:6);

    % target data
    rTM = targetState_M(1:3);
    rTE = rTM + rem;
    vTM = targetState_M(4:6);

    % compute norms
    rTMn = norm(rTM);
    vTMn = rTM'*vTM/norm(rTM); 
    rTEn = norm(rTE);

    % LVLH versors (RVH convention) and Rotation Matrix
    [eR_x, eV_y, eH_z] = versorsLVLH(targetState_M,param); 
    RotMat_M_to_L = [eR_x(:).';eV_y(:).';eH_z(:).'];

    % relative stuff (in LVLH)
    rho = relativeState_L(1:3);
    v_rho = relativeState_L(4:6);

    % computing aMT from the CR3BP from Franzini (Moon Centered)
    dSt = CR3BP_MoonFrame(t,targetState_M,param);
    aTM = dSt(4:6);
    
    % computation of angular momentum and derivatives
    hTM = cross(rTM,vTM);
    hTM_norm = norm(hTM);
    hTM_dot = cross(rTM, aTM);
    hTM_dot_norm = hTM'*hTM_dot/hTM_norm;

    % ANGULAR VELOCITY COMPUTATION

    % M01:
    omegaLM     = [rTMn/hTM_norm*(aTM'*eH_z);
                   0;
                   1/rTMn*(vTM'*eV_y)];

    %% compute jerk from [T14]
    JI = -massRatio*derivataStrana(rTM)*(vTM+cross(omegaMI,rTM)) - (1-massRatio)*( derivataStrana(rTE)*( vTM+cross(omegaMI,rTM)+cross(omegaMI,rem) ) - derivataStrana(rem)*cross(omegaMI,rem) ); %
    JTM = JI - 3*cross(omegaMI,aTM) - 3*cross(omegaMI,cross(omegaMI,vTM)) - cross(omegaMI,cross(omegaMI,cross(omegaMI,rTM)));

    %% ANGULAR ACCELERATIOON

    % M01
    omegaLM_dot = [rTMn/hTM_norm*( vTMn/rTMn*(aTM'*eH_z) - 2*rTMn/hTM_norm*(aTM'*eV_y)*(aTM'*eH_z) + JTM'*eH_z );
                   0;
                   1/rTMn*( aTM'*eV_y - 2*vTMn/rTMn*(vTM'*eV_y) )];

    omegaLI = omegaLM + RotMat_M_to_L*omegaMI;
    omegaLI_dot = omegaLM_dot - cross(omegaLM,RotMat_M_to_L*omegaMI); % omegaLS_dot shall be computed wrt LVLH


    %% RELATIVE ACCELERATION
    rTM =  RotMat_M_to_L*rTM;
    rTE =  RotMat_M_to_L*rTE;
    rCM =  RotMat_M_to_L*rCM;
    rCE =  RotMat_M_to_L*rCE;
    a_rho_MI = massRatio*(rTM/rTMn^3-rCM/norm(rCM)^3) + (1-massRatio)*(rTE/rTEn^3-rCE/norm(rCE)^3);
    a_rho = a_rho_MI - 2*cross(omegaLI,v_rho) - cross(omegaLI_dot,rho) - cross(omegaLI,cross(omegaLI,rho));

    if nargin > 5
        a_rho = a_rho + controlAction_L;
    end
    % RHS of state space dynamics equation of motion
    dRelState = [v_rho(:); a_rho(:)];

    function [der] = derivataStrana(q)
        der = 1/norm(q)^3 * ( eye(3)-3*(q*q')/norm(q)^2 );
    end

end