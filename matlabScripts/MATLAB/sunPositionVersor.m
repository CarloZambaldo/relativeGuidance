function [sunVersor] = sunPositionVersor(t,param)
    if ~isfield(param,'sunInitialAngle')
        param.sunInitialAngle = 0;
    end
    Theta = param.sunInitialAngle+1.996437750711854e-07*t;
    sunVersor = [cos(Theta); sin(Theta); 0];
end