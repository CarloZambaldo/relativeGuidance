function [dSimState,controlAction_L] = simEquations(t,fullState,param)
    %   dSimState = [[xT yT zT vxT vyT vzT]'; [xC yC zC vxC vyC vzC]'; [v_rho_L a_rho_L]']
    % this function is the RHS of the whole environment differential
    % equation, it integrates the dynamics of the 4 bodies (using 2 CR3BPs)
    % and is used to launch the guidance

    % initialisation of the parameters
    targetState_S = fullState(1:6);
    chaserState_S = fullState(7:12);


    %% SPACECRAFT GUIDANCE ALGORITHM
    % compute relative dynamics
    [targetState_M,~,relativeState_L] = OBNavigation(targetState_S,chaserState_S,param);
    %[dRelState_L] = relativeDynamicsLVLH(t,relativeState_L,targetState_M,chaserState_M,param);
    %[dRelState_L] = relativeDynamicsLinearizedLVLH(t,relativeState_L,targetState_M,param);
    dRelState_L = [0;0;0;0;0;0];


    % CONTROL ACTION COMPUTATION
    if ~param.BOOLS.controlActionBool || ~isfield(param.BOOLS,'controlActionBool')
        controlAction_L = [0;0;0];
        controlAction = [0;0;0];
    else
        % [controlAction_L] = APF(convert_S_to_LVLH(targetState_S,chaserState_S-targetState_S,param),param);
        %relativeState_L = fullState(13:18);
        [controlAction_L] = APF(relativeState_L,param);
        %[dRelState_L] = relativeDynamicsLVLH(t,relativeState_L,targetState_M,chaserState_M,param,controlAction_L);
        [controlAction] = rotateControlAction(targetState_S,controlAction_L,param);
    end
    
    %% PROPAGATION OF CR3BP dynamics
    dstateTarget_S = CR3BP(t, targetState_S, param);
    dstateChaser_S = CR3BP(t, chaserState_S, param, controlAction);

    %% full output generations
    dSimState = [dstateTarget_S(:); dstateChaser_S(:); dRelState_L(:)];
end