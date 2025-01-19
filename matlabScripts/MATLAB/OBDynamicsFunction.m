function [dStateOB] = OBDynamicsFunction(relativeState_L,targetState_M,chaserState_M,param)
    %
    % 
    % the "ObBoard Dynamics Function" contains:
    %   - target's dState computation
    %   - relative dState computation
    % 
    % 
    % 

    [dRelState] = relativeDynamicsLVLH(t,relativeState_L,targetState_M,chaserState_M,param);
end