function [distAcceleration] = computeDisturbances(t,SCparam,param)

    if ~isfield(param,'sunInitialAngle')
        param.sunInitialAngle = 0;
    end
    
    Theta = param.sunInitialAngle;%%%%%+1.996437750711854e-07*t;
    sunVersor = [cos(Theta); sin(Theta); 0];


    distAcceleration = (param.SolarFlux*SCparam.Area./SCparam.mass*(1+SCparam.reflCoeffSpecular+2/3*SCparam.reflCoeffDiffuse))*sunVersor;

    % adimensionalize:
    distAcceleration = distAcceleration*(1e-3)*param.tc^2/param.xc;

end