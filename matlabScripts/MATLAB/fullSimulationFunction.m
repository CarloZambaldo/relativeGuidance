function [timeHistory,fullStateHistory,controlActionHistory_L,info] = fullSimulationFunction(fullInitialState,param)
    %     targetState_S = fullInitialState(1:6)';
    %    chaserState_S = fullInitialState(7:12)';


    global trigger OBoptimalTrajectory

    odeopts = odeset("AbsTol",1e-7,"RelTol",1e-8);

    % definition of the solution vectors
    timeHistory = param.tspan(1):(1/param.freqGNC):param.tspan(2);
    controlActionHistory_L = zeros(length(timeHistory)+1,3);
    fullStateHistory = zeros(length(timeHistory),12);
    
    % extraction of the initial conditions
    targetState_S = fullInitialState(1:6)';
    chaserState_S = fullInitialState(7:12)';
    
    fullStateHistory(1,:) = fullInitialState(:)';
    
    % integrate the dynamics of the target
    [distAcceleration_S] = computeDisturbances(0,param.target,param);
    [~,odesol] = ode113(@(t,state)CR3BP(t, state, param, [0;0;0], distAcceleration_S),timeHistory,targetState_S,odeopts);
    fullStateHistory(:,1:6) = odesol;
    
    %% simulation loop (to enforce 10Hz GNC frequency)
    indx = 0;
    aimReachedBool = 0;
    crashedBool = 0;
    
    switch(param.phaseID)
        case 1 %% REACHING SAFE HOLD
            aimAtState = param.holdingState;
        case 2 %% DOCKING PHASE
            aimAtState = param.dockingState;
        otherwise
            error("Wrong phase ID");
    end
    
    while (indx < length(timeHistory)-1) && ~aimReachedBool && ~crashedBool
        indx = indx+1;
        timeNow = timeHistory(indx);
        targetState_S = fullStateHistory(indx,1:6)';
        chaserState_S = fullStateHistory(indx,7:12)';
    
        % NAVIGATION
        [OBStateTarget_M,~,OBStateRelative_L] = OBNavigation(targetState_S,chaserState_S,param);
    
        % GUIDANCE
        tic
        [OBControlAction_L] = OBGuidance(timeNow,OBStateRelative_L,OBStateTarget_M,param.phaseID,param);
        controlActionHistory_L(indx+1,:) = OBControlAction_L(:)';
        %fprintf("       > Guidance Total Exec Time: %.2f ms\n",toc*1e3);
    
        % PROPAGATE SYSTEM DYNAMICS
        [controlAction_S] = rotateControlAction(targetState_S,controlActionHistory_L(indx,:)',param);
    
        [distAcceleration_S] = computeDisturbances(timeNow,param.chaser,param);
        [~,odesol] = ode113(@(t,state) CR3BP(t, chaserState_S, param, controlAction_S, distAcceleration_S),[timeNow timeHistory(indx+1)],chaserState_S,odeopts);
        fullStateHistory(indx+1,7:12) = odesol(end,:);
    
        TRUE_relativeState_L = convert_S_to_LVLH(targetState_S,chaserState_S-targetState_S,param);
        [aimReachedBool,crashedBool] = checkAimReached(TRUE_relativeState_L, aimAtState, param);
    
    end
    info.crashedBool = crashedBool;
    info.aimReachedBool = aimReachedBool;

    %if crashedBool
    %    fprintf("CRASHED.\n");
    %end
    %if aimReachedBool
    %    fprintf(" DOCKING SUCCESSFUL.\n")
    %end
    if ~crashedBool && ~aimReachedBool
        fprintf(" SIMULATION RUN OUT OF TIME . \n");
    end
end