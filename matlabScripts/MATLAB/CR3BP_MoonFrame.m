function [dRelState] = CR3BP_MoonFrame(t, state_M, param)
    % this function computes the derivative of the state for the CR3BP,
    % with respect to a reference frame which is Synodic but centered about
    % the moon, for reference see Franzini PhD thesis [T14]

    massRatio = param.massRatio;
    rem = [-1;0;0];
    rei = state_M(1:3)+rem;
    rmi = state_M(1:3);
    vmi = state_M(4:6);
    omegaMI = [0;0;1];
    
    amiIner = -massRatio*rmi/norm(rmi)^3 - (1-massRatio)*(rei/norm(rei)^3 - rem);
    ami = amiIner - 2*cross(omegaMI,vmi) - cross(omegaMI,cross(omegaMI,rmi));

    dRelState = [vmi(:);ami(:)];
end