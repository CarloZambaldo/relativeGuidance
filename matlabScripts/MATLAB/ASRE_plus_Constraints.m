function optimalTrajectory = ASRE_plus_Constraints(TOF, initialRelativeState_L, initialStateTarget_M, finalAimState, phaseID, param)
    tic
    % PARAMETERS
    t_i = 0;                   % Initial time
    t_f = TOF;                 % Final time
    N = 210;                   % Number of time steps

    % TIME GRID
    tvec = linspace(t_i, t_f, N);
    
    % INITIAL AND FINAL STATES
    x_i = initialRelativeState_L;   % Initial relative state
    x_f = finalAimState(:);
    u_guess = zeros(3, N-1); % Initial control guess (zeros)

    % COST MATRICES
    switch phaseID
        case 1
            Q = eye(6);  % State cost matrix
            R = eye(3);               % Control cost matrix
            
        case 2
            % original
            % Q = [[1e12 0 0; 0 1e-5 0; 0 0 1e12], zeros(3); zeros(3), [1 0 0; 0 1e-6 0; 0 0 1]];     % State cost matrix
            % R = [.9e-1 0 0; 0 1 0; 0 0 .9e-1];                                                      % Control cost matrix 

            % Q = [[1e10 0 0; 0 1e-5 0; 0 0 1e10], zeros(3); zeros(3), [1e-2 0 0; 0 1e3 0; 0 0 1e-2]];   % State cost matrix
            % R = [2e1 0 0; 0 8e1 0; 0 0 2e1];                                                           % Control cost matrix   
            % 
            % %% BEST ONE
            % Q = [[5e4 0 0; 0 5e0 0; 0 0 5e4], zeros(3); zeros(3), [1e6 0 0; 0 1e6 0; 0 0 1e6]];      % State cost matrix
            % R = [2e1 0 0; 0 3e3 0; 0 0 2e1];                                                         % Control cost matrix  


            %% LAST USED
            % Q = [[5e5 0 0; 0 1e2 0; 0 0 5e5], zeros(3); zeros(3), [5e6 0 0; 0 5e6 0; 0 0 5e6]];      % State cost matrix
            % R = [2e1 0 0; 0 2e1 0; 0 0 2e1];                                                         % Control cost matrix  

            %% NOW TESTING 
            Q = [[8e5 0 0; 0 1e2 0; 0 0 8e5], zeros(3); zeros(3), [5e6 0 0; 0 5e6 0; 0 0 5e6]];      % State cost matrix
            R = [2e1 0 0; 0 2e1 0; 0 0 2e1];                                                         % Control cost matrix  
            W = diag(1,1,1);

        otherwise
            Q = eye(6);               % State cost matrix
            R = eye(3);               % Control cost matrix  
    end

    % INITIALIZE ITERATION
    x_guess = interpolateTrajectory(x_i, x_f, tvec); % Linear initial guess

    %% ITERATIVE ASRE PROCESS

    % iteration 0
    A = computeA(initialStateTarget_M, param);
    B = computeB();
    
    phi_xx = eye(6);
    phi_yy = eye(6);
    phi_xy = zeros(6);
    phi_yx = zeros(6);

    PHI0 = [phi_xx, phi_xy; phi_yx, phi_yy];
    % % [~,PHI]  = ode78(@(t,PHI)computePHI(t,PHI,A,B,Q,R),[tvec(1) tvec(end)],PHI0);
    % % PHI = reshape(PHI(end,:),12,12);
    M12 = -B*(R\B');
    [~,PHIT]  = ode78(@(t,PHIT)computePHIT(t,PHIT,B,Q,R,M12,param),tvec,[reshape(PHI0,144,1);initialStateTarget_M]);
    PHI = PHIT(end,1:144);
    PHI = reshape(PHI(end,:),12,12);

    lambda_i =  PHI(1:6,7:12)\(x_f-PHI(1:6,1:6)*x_i);

    x_new(:,1) = x_i;
    u_new(:,1) = -R\(B'*lambda_i);

    % for each time step (compute the trajectory)
    for time_id = 2:N

        %[~,PHIT]  = ode78(@(t,PHIT)computePHIT(t,PHIT,B,Q,R,param),[tvec(1) tvec(time_id)],[reshape(PHI0,144,1);initialStateTarget_M]);
        PHI = PHIT(time_id,1:144);
        PHI = reshape(PHI(end,:),12,12);

        lambda = PHI(7:12,1:6)*x_i + PHI(7:12,7:12)*lambda_i;
        x_new(:,time_id) = PHI(1:6,1:6)*x_i + PHI(1:6,7:12)*lambda_i;
        u_new(:,time_id) = -R\(B'*lambda);
    end

    % Update the guess for the next iteration
    x_guess = x_new;
    u_guess = u_new;

    %    fprintf('Iteration %d: Max Error = %g\n', iteration, epsilon);
    % end

    % OUTPUT OPTIMAL TRAJECTORY
    optimalTrajectory.t = tvec;
    optimalTrajectory.x = x_guess;
    optimalTrajectory.u = u_guess;

    exectime = toc;
    fprintf('ASRE Converged. [Execution time: %f sec]\n',exectime);
end


function x_guess = interpolateTrajectory(x_i, x_f, tvec)
    % Generates an initial guess for the state trajectory as a linear interpolation.
    x_guess = zeros(length(x_i), length(tvec));
    for i = 1:length(x_i)
        x_guess(i, :) = linspace(x_i(i), x_f(i), length(tvec));
    end
end

function A = computeA(targetState_M, param)
    % Computes state-dependent dynamics matrix A(state).
    [A] = relDynOBmatrixA(1,targetState_M,param);
end

function B = computeB()
    % Computes control influence matrix B(state).
    B = zeros(6, 3); 
    B(4:6, :) = eye(3);
end


function [DStatePHIT] = computePHIT(t,PHIT,B,Q,R, M12,param)
    PHI = PHIT(1:144);
    targetState_M = PHIT(145:150);
    
    % target state and system dynamics retrieval
    dST = CR3BP_MoonFrame(t,targetState_M,param);
    [A] = relDynOBmatrixA(t,targetState_M,param);
    %Q = computeQ(t,targetState_M,param);
    % PHI computation
    PHI = reshape(PHI,12,12);
    %%% M = [A, -B*(R\B'); -Q, -A];
    M = [A, M12; -Q, -A];
    DP = M*PHI;
    DP = reshape(DP,12*12,1);

    DStatePHIT = [DP;dST];
end

function [Amat] = relDynOBmatrixA(t,targetState_M,param)
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

    OMEGA_LI = [  0      -omegaLI(3)  omegaLI(2);
                omegaLI(3)    0      -omegaLI(1);
                -omegaLI(2)  omegaLI(1)    0    ];

    OMEGA_LI_dot = [  0      -omegaLI_dot(3)  omegaLI_dot(2);
                    omegaLI_dot(3)    0      -omegaLI_dot(1);
                    -omegaLI_dot(2)  omegaLI_dot(1)    0    ];

    Xi = -massRatio/rTMn^3*(eye(3)-3*(rTM*rTM')/rTMn^2) - (1-massRatio)/rTEn^3*(eye(3)-3*(rTE*rTE')/rTEn^2);
    Amat = [zeros(3), eye(3); Xi-OMEGA_LI_dot-OMEGA_LI^2, -2*OMEGA_LI];

    function [der] = derivataStrana(q)
        der = 1/norm(q)^3 * ( eye(3)-3*(q*q')/norm(q)^2 );
    end

end