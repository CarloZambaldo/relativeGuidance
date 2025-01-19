function [dSimState,controlAction_L] = guidanceSimEquations(t,fullState,phaseID,param)
    %   dSimState = [[xT yT zT vxT vyT vzT]'; [xC yC zC vxC vyC vzC]']
    % this function is the RHS of the whole environment differential
    % equation, it integrates the dynamics of the 4 bodies (using 2 CR3BPs)
    % and is used to launch the guidance

    global trigger;

    % initialisation of the parameters
    targetState_S = fullState(1:6);
    chaserState_S = fullState(7:12);

    % environment
    [sunVersor] = sunPositionVersor(t,param);

    %% SPACECRAFT GUIDANCE ALGORITHM
    % compute relative dynamics
    [targetState_M,~,relativeState_L] = OBNavigation(targetState_S,chaserState_S,param);

    % CONTROL ACTION COMPUTATION
    [controlAction_L] = OBGuidance(t,relativeState_L,targetState_M,phaseID,param);
    [controlAction] = rotateControlAction(targetState_S,controlAction_L,param);
    
    %% PROPAGATION OF CR3BP dynamics
    dstateTarget_S = CR3BP(t, targetState_S, param);
    dstateChaser_S = CR3BP(t, chaserState_S, param, controlAction);

    %% full output generations
    dSimState = [dstateTarget_S(:); dstateChaser_S(:)];
end