function [targetState_M,chaserState_M,relativeState_L] = OBNavigation(targetState_S,chaserState_S,param)
    %
    % [targetState_M,chaserState_M,relativeState_L] = OBNavigation(targetState_S,chaserState_S,param)
    %
    % this function outputs the translation and rotation from Synodic to
    % Moon centered synodic

    %% physical values
    rM = [1-param.massRatio;0;0]; % position of the moon in Synodic frame

    %% TRASLATING AND ROTATING to MOON CENTERED SYNODIC
    %FranziRot = [[-1 0 0; 0 -1 0; 0 0 1], zeros(3); zeros(3), [-1 0 0; 0 -1 0; 0 0 1]];
    targetState_M = [-targetState_S(1:2)+rM(1:2); targetState_S(3)-rM(3); -targetState_S(4:5); targetState_S(6)]; % is the same as: FranziRot*(targetState_S-[rM(:);0;0;0])
    chaserState_M = [-chaserState_S(1:2)+rM(1:2); chaserState_S(3)-rM(3); -chaserState_S(4:5); chaserState_S(6)]; %FranziRot*(chaserState_S-[rM(:);0;0;0]);

    % dimensionalize
    % targetState_M = [targetState_M(1:3); targetState_M(4:6)./param.tc]*1e3*param.xc;
    % chaserState_M = [chaserState_M(1:3); chaserState_M(4:6)./param.tc]*1e3*param.xc;
    
    %% COMPUTING RELATIVE STATE IN MOON CENTERED SYNODIC AND ROTATE TO LVLH
    if nargout>2
        relativeState_M = chaserState_M-targetState_M;
        relativeState_L = convert_M_to_LVLH(targetState_M,relativeState_M,param);

        % dimensionalize
        % relativeState_L = [relativeState_L(1:3); relativeState_L(4:6)./param.tc]*1e3*param.xc;
    end
end