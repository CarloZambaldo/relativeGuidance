function [controlAction_L] = OBGuidance(envTime,relativeState,targetState,phaseID,param)
% DOCKING PHASE

    % this function is the OB Guidance block
    global trigger OBoptimalTrajectory;
    Umax = param.maxAdimThrust;
    rho = relativeState(1:3);

    switch(phaseID)
        case 1 %% REACHING SAFE HOLD
            constraintType = 'SPHERE';
            aimAtState = param.holdingState;
            characteristicSize = 200;
        case 2 %% DOCKING PHASE
            constraintType = 'CONE';
            aimAtState = param.dockingState;
            characteristicSize.acone = 0.08;
            characteristicSize.bcone = 5;
        otherwise
            error("Wrong phase ID");
    end

    %% LOOP 1
     % computation of the optimal Trajectory
    if trigger % recomputation of loop 1
        OBoptimalTrajectory = loopOne(envTime,relativeState,targetState,aimAtState,phaseID,param);
        if ~isempty(OBoptimalTrajectory)
            constraintViolationFlag = checkConstraintViolation(OBoptimalTrajectory,constraintType,characteristicSize);
            if constraintViolationFlag
                warning("CONSTRAINTS COULD BE VIOLATED WITH THE GIVEN TRAJECTORY.")
            end
        end
    end

    %% LOOP 2
   
    % compute the APF surface
    [closestOptimalControl,surface_L1_pos,surface_L1_vel,surface_L2] = loopTwo(envTime,relativeState,aimAtState,OBoptimalTrajectory,constraintType,param);

    % sliding surface computation
    sigma = surface_L2 + (5*surface_L1_vel + 6e-3*surface_L1_pos);
    %       ^ APF REP ^        ^  OPTIMAL TRAJECTORY VEL + POS  ^    %

    %% adding the terms inside the SMC 
    controlAction_L = (closestOptimalControl - Umax*tanh(sigma)); % original
    % residualThrust = (1-abs(tanh(surface_L2)));
    % controlAction_L = (residualThrust.*closestOptimalControl - Umax.*tanh(sigma));
    %                          ^ OPTIMAL CONTROL ACTION ^          ^  SMC  ^

end



%% %% %% %% %% %%
%%  FUNCTIONS  %%
%% %% %% %% %% %%

%% L1
function [optimalTrajectory] = loopOne(envTime,initialRelativeState_L,initialTargetState_M,aimAtState,phaseID,param)
    %fprintf("Computing Optimal Trajectory... "); tic
    %TOF = norm(initialRelativeState_L(1:3))/(1e-3/param.xc*param.tc)*1.1;
    TOF = computeTOF(initialRelativeState_L,aimAtState,param);
    if TOF > 0%1e-5
        %fprintf("\n  Estimated TOF: %f [-]\n",TOF);
        optimalTrajectory = ASRE(TOF,initialRelativeState_L,initialTargetState_M,aimAtState,phaseID,param);
        optimalTrajectory.envStartTime = envTime;
        %fprintf(" done. [Elapsed Time: %.3f sec]\n",toc);

    else
        fprintf("\n  Estimated TOF is too small. OBoptimalTrajectory is set to empty.\n");
        optimalTrajectory = [];
    end
    % % % optimalTrajectory = [];  % TO HAVE STD SMC+APF ONLY% % % % % 
    global trigger
    trigger = 0;
end


%% L2
function [closestOptimalControl,surface_L1_pos,surface_L1_vel,surface_L2] = loopTwo(envTime,relativeState,aimAtState,OBoptimalTrajectory,constraintType,param)
    % NOTE: the control action is in adimensional units
    global trigger

    % if the optimal trajectory exists add it to the surface
    if ~isempty(OBoptimalTrajectory)
        interpTime = (envTime-OBoptimalTrajectory.envStartTime);
        if interpTime < 0
            warning("Error in time definition. This is possibly due to the numerical integrator\n");
            interpTime = OBoptimalTrajectory.t(end);
            if interpTime < -1e-7
                error("COULD NOT PROCEED.");
            end
        end
    else
        interpTime = NaN;
    end
    if ~isempty(OBoptimalTrajectory) && isfield(OBoptimalTrajectory,"x") && OBoptimalTrajectory.t(end)>=interpTime 
        closestOptimalState(1,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(1,:),interpTime);
        closestOptimalState(2,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(2,:),interpTime);
        closestOptimalState(3,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(3,:),interpTime);
        closestOptimalState(4,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(4,:),interpTime);
        closestOptimalState(5,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(5,:),interpTime);
        closestOptimalState(6,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.x(6,:),interpTime);

        closestOptimalControl(1,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.u(1,:),interpTime);
        closestOptimalControl(2,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.u(2,:),interpTime);
        closestOptimalControl(3,1) = interp1(OBoptimalTrajectory.t,OBoptimalTrajectory.u(3,:),interpTime);

        %fprintf("  [envTime %f] closestOptimalState [dist = %f m; |deltaV| = %f m/s] \n",envTime,norm(relativeState(1:3)-closestOptimalState(1:3))*1e3*param.xc,norm(closestOptimalState(4:6)-relativeState(4:6))*1e3*param.xc/param.tc);
        trigger = 0;
    else
        closestOptimalState = aimAtState;
        closestOptimalControl = [0;0;0];

        %fprintf("  [envTime %f] <info> Using aimAtState as convergence point [dist = %f m; |deltaV| = %f m/s] \n",envTime,norm(relativeState(1:3)-closestOptimalState(1:3))*1e3*param.xc,norm(closestOptimalState(4:6)-relativeState(4:6))*1e3*param.xc/param.tc);
        % trigger = 1;
    end
    %fprintf("   goal distance: %f m; |deltaV| = %f m/s]", norm(relativeState(1:3)-aimAtState(1:3))*1e3*param.xc, norm(relativeState(4:6)-aimAtState(4:6))*1e3*param.xc/param.tc )

    surface_L1_pos = (relativeState(1:3)-closestOptimalState(1:3))*1e3*param.xc; %  
    surface_L1_vel = (relativeState(4:6)-closestOptimalState(4:6))*1e3*param.xc/param.tc;

    [~,surface_L2] = APF(relativeState,constraintType,param);
end