function [dRelState,Amat] = relativeDynamicsLinearizedLVLH(t,relativeState_L,targetState_M,param)
    %   
    % [dRelState,Amat] = relativeDynamicsLinearizedLVLH(t,relativeState_L,targetState_M,param)
    %
    % this function outputs the RHS of the relative dynamics equation found
    % in Franzini work, this is the LINEARIZED equation

    %% EXTRACTING VALUES FROM INPUT
    % environment data [in Moon Synodic]
    massRatio = param.massRatio;
    omegaMI = [0;0;1];
    rem = [-1;0;0];

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

    % computing aMT from the CR3BP from Franzini (Moon Centered)
    dSt = CR3BP_MoonFrame(t,targetState_M,param);
    aTM = dSt(4:6);
    
    % computation of angular momentum and derivatives
    hTM = cross(rTM,vTM);
    hTM_norm = norm(hTM);
    hTM_dot = cross(rTM, aTM);
    hTM_dot_norm = hTM'*hTM_dot/hTM_norm;

    % ANGULAR VELOCITY COMPUTATION
    omegaLM     = [rTMn/hTM_norm*(aTM'*eH_z);
                   0;
                   1/rTMn*(vTM'*eV_y)];

    % compute jerk from [T14]
    JI = -massRatio*derivataStrana(rTM)*(vTM+cross(omegaMI,rTM)) - (1-massRatio)*( derivataStrana(rTE)*( vTM+cross(omegaMI,rTM)+cross(omegaMI,rem) ) - derivataStrana(rem)*cross(omegaMI,rem) ); %
    JTM = JI - 3*cross(omegaMI,aTM) - 3*cross(omegaMI,cross(omegaMI,vTM)) - cross(omegaMI,cross(omegaMI,cross(omegaMI,rTM)));

    % ANGULAR ACCELERATION
    omegaLM_dot = [rTMn/hTM_norm*( vTMn/rTMn*(aTM'*eH_z) - 2*rTMn/hTM_norm*(aTM'*eV_y)*(aTM'*eH_z) + JTM'*eH_z );
                   0;
                   1/rTMn*( aTM'*eV_y - 2*vTMn/rTMn*(vTM'*eV_y) )];

    omegaLI = omegaLM + RotMat_M_to_L*omegaMI;
    omegaLI_dot = omegaLM_dot - cross(omegaLM,RotMat_M_to_L*omegaMI); % omegaLS_dot shall be computed wrt LVLH


    %% RELATIVE ACCELERATION
    rTM =  RotMat_M_to_L*rTM;
    rTE =  RotMat_M_to_L*rTE;
    % rCM =  rTM + rho;
    % rCE =  rTM + rho + rem;

    OMEGA_LI = [  0      -omegaLI(3)  omegaLI(2);
                omegaLI(3)    0      -omegaLI(1);
                -omegaLI(2)  omegaLI(1)    0    ];

    OMEGA_LI_dot = [  0      -omegaLI_dot(3)  omegaLI_dot(2);
                    omegaLI_dot(3)    0      -omegaLI_dot(1);
                    -omegaLI_dot(2)  omegaLI_dot(1)    0    ];

    Xi = -massRatio/rTMn^3*(eye(3)-3*(rTM*rTM')/rTMn^2) - (1-massRatio)/rTEn^3*(eye(3)-3*(rTE*rTE')/rTEn^2);
    Amat = [zeros(3), eye(3); Xi-OMEGA_LI_dot-OMEGA_LI^2, -2*OMEGA_LI];

    % RHS of state space dynamics equation of motion
    dRelState = Amat*relativeState_L;

    function [der] = derivataStrana(q)
        der = 1/norm(q)^3 * ( eye(3)-3*(q*q')/norm(q)^2 );
    end

end