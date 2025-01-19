function [dRelState] = relativeDynamicsNLRESynodic(t,relativeState,chaserState,param,targetState)

    massRatio = param.massRatio;
    if nargin<5
        warning("Estimating target State via relative dynamics")
        targetState = chaserState - relativeState; % relativeState = [x y z vx vy vz]'
    end
    %% NON-LINEAR RELATIVE EQUATIONS IN SYNODIC REFERENCE SYSTEM
    xT  = targetState(1);
    yT  = targetState(2);
    zT  = targetState(3);
    vxT = targetState(4);
    vyT = targetState(5);
    vzT = targetState(6);
    xC  = chaserState(1);
    yC  = chaserState(2);
    zC  = chaserState(3);
    vxC = chaserState(4);
    vyC = chaserState(5);
    vzC = chaserState(6);

    % % C - T (Chaser as seen from Target) -> TARGET IS THE REFERENCE
    % %this is like: ar = ar_Inertial - 2*cross(omegaS,vC-vT) -  cross(omegaS,cross(omegaS,rC-rT)
        arx = 2*(vyC-vyT) + xC - xT - (massRatio*(massRatio + xC - 1))/((massRatio + xC - 1)^2 + yC^2 + zC^2)^(3/2) + (massRatio*(massRatio + xT - 1))/((massRatio + xT - 1)^2 + yT^2 + zT^2)^(3/2) + ((massRatio + xC)*(massRatio - 1))/((massRatio + xC)^2 + yC^2 + zC^2)^(3/2) - ((massRatio + xT)*(massRatio - 1))/((massRatio + xT)^2 + yT^2 + zT^2)^(3/2);
        ary = 2*(vxT-vxC) + yC - yT + yC*((massRatio - 1)/((massRatio + xC)^2 + yC^2 + zC^2)^(3/2) - massRatio/((massRatio + xC - 1)^2 + yC^2 + zC^2)^(3/2)) - yT*((massRatio - 1)/((massRatio + xT)^2 + yT^2 + zT^2)^(3/2) - massRatio/((massRatio + xT - 1)^2 + yT^2 + zT^2)^(3/2));
        arz = zC*((massRatio - 1)/((massRatio + xC)^2 + yC^2 + zC^2)^(3/2) - massRatio/((massRatio + xC - 1)^2 + yC^2 + zC^2)^(3/2)) - zT*((massRatio - 1)/((massRatio + xT)^2 + yT^2 + zT^2)^(3/2) - massRatio/((massRatio + xT - 1)^2 + yT^2 + zT^2)^(3/2));

        dRelState = [relativeState(4:6); arx; ary; arz];

    %% ALTERNATIVE
    % % % A = [zeros(3), eye(3); Xi+[1 0 0; 0 1 0; 0 0 0], - 2*[0 -1 0; 1 0 0; 0 0 0] ];
    % % % dRelState = A*relativeState 

    %% MY NONLINEAR RE (LVLH)
    % x = relativeState(1);
    % y = relativeState(2);
    % z = relativeState(3);
    % 
    % xT  = targetState(1);
    % yT  = targetState(2);
    % zT  = targetState(3);
    % vxT = targetState(4);
    % vyT = targetState(5);
    % vzT = targetState(6);
    % 
    % xC  = chaserState(1);
    % yC  = chaserState(2);
    % zC  = chaserState(3);
    % vxC = chaserState(4);
    % vyC = chaserState(5);
    % vzC = chaserState(6);
    % 
    % rTEn = sqrt((xT+massRatio)^2+yT^2+zT^2);
    % rTMn = sqrt((xT-1+massRatio)^2 + yT^2 + zT^2);
    % 
    % rCEn = sqrt((xC+massRatio)^2+yC^2+zC^2);
    % rCMn = sqrt((xC-1+massRatio)^2 + yC^2 + zC^2);
    % 
    % vx = relativeState(4);
    % vy = relativeState(5);
    % vz = relativeState(6);
    % dRelState = [vx, vy, vz, ax, ay, az];

end