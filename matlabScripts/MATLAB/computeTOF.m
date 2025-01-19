function TOF = computeTOF(relativeState,aimAtState,param)
    ri = relativeState(1:3);
    vi = relativeState(4:6);
    
    
    % vdelta = (vi'*delta)/(delta'*delta)*delta;

    % compute the norm
    % vperpenorm = norm(vi-vdelta);
    % vdeltavalue = (vi'*delta)/norm(delta); % with Sign!!

    % TOF = norm(delta)/norm(vdelta)*2e-1
    % TOF = 2*(norm(delta)/(6e-4) + vperpenorm/.01 - vdeltavalue/.05 + norm([delta(1);delta(3)])/(.5e-3));

    delta = ri - aimAtState(1:3);
    p_factor = 2 + delta(2)./norm(delta);
    o_factor = 1.1 - tanh(norm(delta)*param.xc/5);
    TOF = norm(delta)/5e-4 * o_factor * p_factor;

    fprintf("Estimated TOF: %f (%f [hours])",TOF,TOF*param.tc/3600)
end