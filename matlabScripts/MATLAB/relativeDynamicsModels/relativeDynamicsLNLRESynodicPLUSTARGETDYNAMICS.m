function [dRelState] = relativeDynamicsLNLRESynodicPLUSTARGETDYNAMICS(t,relativeState,targetState,param)

    massRatio = param.massRatio;
    
    dstateTarget_S = CR3BP(t, targetState, param);

    %% LINEARIZED NON-LINEAR RELATIVE EQUATIONS IN INERTIAL FRAME (TARGET AS REFERENCE)
    rTE = targetState(1:3) - [-massRatio; 0; 0];
    rTM = targetState(1:3) - [1-massRatio; 0; 0];
    eTE = rTE/norm(rTE);
    eTM = rTM/norm(rTM);
    c1 = (1-massRatio)/(norm(rTE))^3;
    c2 = massRatio/(norm(rTM))^3;
    Xi = -(c1+c2)*eye(3) + 3*c1*(eTE*eTE') + 3*c2*(eTM*eTM');
    ar = (Xi+[1 0 0; 0 1 0; 0 0 0])*relativeState(1:3) - 2*[0 -1 0; 1 0 0; 0 0 0]*relativeState(4:6);
    dRelState = [relativeState(4:6); ar(:); dstateTarget_S];
end