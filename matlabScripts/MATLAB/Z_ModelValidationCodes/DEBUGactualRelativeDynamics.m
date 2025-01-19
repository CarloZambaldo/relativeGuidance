function [dSimState] = DEBUGactualRelativeDynamics(t,fullState,param)
    % here add all the 
    %   dSimState = [xT yT zT vxT vyT vzT xC yC zC vxC vyC vzC]
    %

    % initialisation of the parameters
    targetState = fullState(1:6);
    chaserState = fullState(7:12);
    relativeState = fullState(13:18);

    [dRelState] = relativeDynamicsNLRESynodic(t,relativeState,chaserState,param,targetState);


    %% PROPAGATION OF CR3BP dynamics
    dstateTarget = CR3BP(t, targetState, param);

    % if param.guidanceBool
    %     dstateChaser = CR3BP(t, chaserState, param, uTOT);
    % else
    dstateChaser = CR3BP(t, chaserState, param);

    %% full output generations
    dSimState = [dstateTarget(:); dstateChaser(:); dRelState(:)];
end