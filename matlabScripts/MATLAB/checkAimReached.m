function [aimReachedBool,crashedBool] = checkAimReached(TRUE_relativeState_L, aimAtState, param)
    crashedBool = 0;
    aimReachedBool = 0;
    % if the position converges
    if (norm(TRUE_relativeState_L(2)-aimAtState(2)) <= 1.3007e-10 ) % when below 5 cm along V-BAR check if converged:
        if (norm(TRUE_relativeState_L(1:2:3)-aimAtState(1:2:3)) <= 2.6015e-10 ) % stop when below 10 cm error (5cm = 1.3007e-10)
            % if also the velocity converges
            % docking standard: along R and H: max 0.04 m/s; along V: max 0.1 m/s
            if  (  abs(TRUE_relativeState_L(4)-aimAtState(4)) <= 3.9095e-05 ...
                && abs(TRUE_relativeState_L(6)-aimAtState(6)) <= 3.9095e-05 ...
                && abs(TRUE_relativeState_L(5)-aimAtState(5)) <= 9.7737e-05  )
    
                aimReachedBool = 1;
                fprintf("\n <!>>>>> DOCKING SUCCESSFUL <<<<<<!> \n\n")
            else
                crashedBool = 1;
            end
        elseif param.phaseID == 2 && ((TRUE_relativeState_L(2)-aimAtState(2)) > 0) % stop when in front of the target
            crashedBool = 1;
        end
    end
end