function [violationFlag,violationPosition] = checkConstraintViolation(trajectory,contraintType,characteristicSize)
    % 
    violationFlag = false;
    
    if isfield(trajectory,"x")
        if isfield(trajectory,"u")
            controlAction = trajectory.u;
            violationPosition = zeros(2,length(controlAction));
            for idx = 1:length(controlAction)
                if norm(controlAction(:,idx)) > 12
                    violationFlag = true;
                    violationPosition(2,idx) = 1;
                    %warning("Violation of Thrust Constraint");
                end
            end
        end
        trajectory = trajectory.x;
    else
        trajectory = trajectory.x;
        violationPosition = zeros(1,size(trajectory,2));
    end
    if size(trajectory,1)>6
        trajectory = trajectory';
    end
    switch upper(contraintType)
        case 'SPHERE'
            for idx=1:size(trajectory,2)
                if trajectory(1,idx)^2+trajectory(2,idx)^2+trajectory(3,idx)^2 <= characteristicSize^2
                    violationFlag = true;
                    violationPosition(1,idx) = 1;
                end
            end
        case 'CONE'
            for idx=1:size(trajectory,2)
                if  characteristicSize.acone^2*(trajectory(2,idx)-characteristicSize.bcone)^3 + trajectory(1,idx)^2 + trajectory(3,idx)^2 > 0
                    violationFlag = true;
                    violationPosition(1,idx) = 1;
                end
            end
    end
    
    if violationFlag
        warning("The computed Trajectory violates the constraints.");
    else
        fprintf("No violations of the constraints identified.\n");
    end

end