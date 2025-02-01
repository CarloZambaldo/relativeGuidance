function dstate = CR3BP(t, state, param, controlAction, disturbAction)
    %  dstate = CR3BP(t, state, param)
    %  
    %   massRatio = m2/(m1+m2)

    if nargin < 4 % if a control action is not present just set it to zero
        controlAction = [0 0 0]';
    end
    if nargin < 5
        disturbAction = [0;0;0];
    end
    
    x = state(1);
    y = state(2);
    z = state(3);
    massRatio = param.massRatio;

    r1 = sqrt((x+massRatio)^2+y^2+z^2);
    r2 = sqrt((x-1+massRatio)^2 + y^2 + z^2);

    % EQUATIONS OF MOTION
    vx = state(4);
    vy = state(5);
    vz = state(6);
    ax = 2*vy + x - (1-massRatio)/r1^3*(x+massRatio) - massRatio/r2^3*(x-1+massRatio) + controlAction(1) + disturbAction(1);
    ay = -2*vx + y - y*((1-massRatio)/r1^3 + massRatio/r2^3) + controlAction(2) + disturbAction(2);
    az = -z*((1-massRatio)/r1^3 + massRatio/r2^3) + controlAction(3) + disturbAction(3);

    % DIFFERENTIAL OF STATE
    dstate = [vx vy vz ax ay az]';

end