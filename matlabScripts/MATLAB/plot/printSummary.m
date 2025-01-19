% calcoli
[~,~,relativeState_L] = OBNavigation(initialStateTarget_S,initialStateChaser_S,param);
if relativeState_L(2) <= 0
    rP = "BEHIND";
else
    rP = "IN FRONT OF";
end
gg = (relativeState_L(4:6)'*relativeState_L(1:3)./norm(relativeState_L(4:6))/norm(relativeState_L(1:3)));
if gg > 0
    rV = "AWAY FROM";
else
    rV = "TOWARDS";
end
anglo = acos(gg);

relDynami = zeros(length(adimensionalSolution.x),6);
soluz = adimensionalSolution.y';
for id = 1:length(adimensionalSolution.x) 
    [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
    relDynami(id,1:6) = rotatedRelativeState;
end

controlAction = adimensionalSolution.controlAction;
relDynami = relDynami.*param.xc;
switch phaseID
    case 1
        positionError = norm(relDynami(end,1:3)'-param.holdingState(1:3)*param.xc)*1e3;
        velocityError = norm(relDynami(end,4:6)'-param.holdingState(4:6)*param.xc)*1e3/param.tc;
    case 2
        positionError = norm(relDynami(end,1:3)'-param.dockingState(1:3)*param.xc)*1e3;
        velocityError = norm(relDynami(end,4:6)'-param.dockingState(4:6)*param.xc)*1e3/param.tc;
end

%% PRINTS
fprintf("#############################\n")

fprintf("\nPhaseID: %d\n",phaseID);
fprintf("Seed used: %s\n", num2str(param.rng_settings.Seed))
fprintf("Simulated Time: %.2f [hours]\n",adimensionalSolution.x(end)/3600*param.tc);
if ~isempty(OBoptimalTrajectory)
    fprintf(" Optimal Trajectory TOF estimated: %.2f [hours]\n",OBoptimalTrajectory.t(end)/3600*param.tc)
end
fprintf("Initial Distance between C and T: %.2f [km]\n",norm(DeltaIC_S(1:3))*param.xc);
fprintf("Initial Relative velocity between C and T: %.2f [m/s]\n",norm(DeltaIC_S(4:6))*param.xc/param.tc*1e3);
fprintf("Initially: Chaser is *%s* the Target and moving *%s* the Target at an angle of %.2f [deg]\n",rP,rV,rad2deg(anglo));

fprintf("\n#############################\n\n")

fprintf("Maximum Thrust Available: %g [-]\n",param.maxAdimThrust);
fprintf("Maximum Thrust Required : %g [-]\n",max(max(controlAction)));
fprintf("Maximum Thrust Required (norm) : %g [-]\n\n",max(norm(controlAction,1)));

fprintf("Actual Final Position Error:    %g [m]\n",positionError);
fprintf("Actual Final Velocity Error:    %g [m/s]\n",velocityError);

fprintf("\n#############################\n\n")

fprintf("TERMINAL STATE:\n")
fprintf("  position R: %f [m]\n",  relDynami(end,1)*1e3)
fprintf("  position V: %f [m]\n",  relDynami(end,2)*1e3)
fprintf("  position H: %f [m]\n",  relDynami(end,3)*1e3)
fprintf("  velocity R: %f [m/s]\n",relDynami(end,4)*1e3/param.tc)
fprintf("  velocity V: %f [m/s]\n",relDynami(end,5)*1e3/param.tc)
fprintf("  velocity H: %f [m/s]\n",relDynami(end,6)*1e3/param.tc)

fprintf("\n");
if info.aimReachedBool
    fprintf("\n <!>>>>> DOCKING SUCCESSFUL <<<<<<!> \n\n")
elseif info.crashedBool
    fprintf("\n <!!!!><><><> CRASHED <><><><!!!!> \n\n")
else
    fprintf("\n SIMULATION RUN OUT OF TIME \n")
end
fprintf("\n#################################\n")
fprintf("############## END ##############\n")
fprintf("#################################\n")