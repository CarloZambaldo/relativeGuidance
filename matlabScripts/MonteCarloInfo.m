function [meanFinalState,sigmaFinalState] = MonteCarloInfo(data)

    phaseID = data.phaseID;
    param = data.param;
    n_population = data.n_population;
    timeHistory = data.timeHistory;
    %trajectory = data.trajectory;
    trueRelativeStateHistory_L = data.trueRelativeStateHistory_L;
    controlAction = data.controlAction;
    OBoTUsage = [zeros(1,n_population); data.OBoTUsage];
    AgentAction = data.AgentAction; % Azione agente: (timestep, n_simulation)
    fail = data.fail;
    success = data.success;
    terminalState = data.terminalState;
    endTimeIx = data.terminalTimeIndex+1;

    if isfield(data,"CPUExecTimeHistory")
        execTime = data.CPUExecTimeHistory;
    else
        execTime = 0;
    end
    
    if isfield(data,"agentModelName")
        agentModelName = data.agentModelName;
    else
        agentModelName = "N/A";
    end
    terminalState_dimensional = terminalState*param.xc*1e3;
    terminalState_dimensional(4:6,:) = terminalState_dimensional(4:6,:)/param.tc;

    fprintf("\n===========================================================\n")
    %% general info
    fprintf("-- GENERAL INFO --\n")
    fprintf("data contains:    %d [simulations]\n", n_population);
    fprintf("Simulated Phase:  %d\n",phaseID)
    fprintf("Agent model used: %s\n", agentModelName);

    fprintf("\n")
    %%
    fprintf("-- DATA STATISTICS --\n")
    successRate = sum(success)/double(n_population)*100;
    fprintf("SUCCESS RATE:  %3.2f %%\n",successRate);

    failRate = sum(fail)/double(n_population)*100;
    fprintf("FAIL RATE:     %3.2f %%\n",failRate);

    notConverged = 100 - failRate - successRate;
    fprintf("NOT CONVERGED: %3.2f %%\n",notConverged);
    
    fprintf("\n")


    %% COMPUTE STUFF
    meanControl_dim = zeros(1,3); % THIS IS IN NEWTON!!
    totalImpulse = zeros(1,3);
    totalMassUsed = 0;
    simulationMass = zeros(n_population,1);
    
    for sim_id = 1:n_population  
        time_dim = timeHistory(1:endTimeIx(sim_id)) .* param.tc;
        dt = 1/param.freqGNC .* param.tc;

        control_dim = double(param.chaserMass) .* controlAction(1:endTimeIx(sim_id),:,sim_id) .* param.xc * 1e3 ./ param.tc^2;
        
        meanControl_dim = meanControl_dim + trapz(time_dim,control_dim,1);

        totalImpulse = totalImpulse + sum(control_dim(1:end-1,:) .* dt, 1);
        simulationMass(sim_id) = sum(vecnorm(control_dim(1:end-1,:), 2, 1) .* dt) / double(param.chaserSpecificImpulse) / 9.81;
    end

    [sigmaICp, meanICp] = std(vecnorm(trueRelativeStateHistory_L(1,1:3,:),2,2)*param.xc);
    [sigmaICv, meanICv] = std(vecnorm(trueRelativeStateHistory_L(1,4:6,:),2,2)*param.xc/param.tc);
    [sigmaTOF, meanTOF] = std(timeHistory(endTimeIx));
    [sigmaMass, meanMass] = std(simulationMass);
    [sigmaExecTime, meanExecTime] = std(execTime(execTime~=0));
    meanControl_dim = meanControl_dim / double(n_population);
    totalImpulse = totalImpulse / double(n_population);
    



    %% STATISTICS ON THE FINAL POSITION
    [sigmaFinalState, meanFinalState] = std(terminalState_dimensional,0,2);
    fprintf("Final Position R-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(1)*1e2, sigmaFinalState(1)*1e2);
    fprintf("Final Position V-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(2)*1e2, sigmaFinalState(2)*1e2);
    fprintf("Final Position H-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(3)*1e2, sigmaFinalState(3)*1e2);
    fprintf("Final Velocity R-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(4)*1e2, sigmaFinalState(4)*1e2);
    fprintf("Final Velocity V-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(5)*1e2, sigmaFinalState(5)*1e2);
    fprintf("Final Velocity H-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(6)*1e2, sigmaFinalState(6)*1e2);
    

    %% MEAN THRUST REQUIRED
    fprintf("\n-- MEAN CONTROL ACTION --\n");
    fprintf("Mean Control Action R = %.2f [N]\n", meanControl_dim(1));
    fprintf("Mean Control Action V = %.2f [N]\n", meanControl_dim(2));
    fprintf("Mean Control Action H = %.2f [N]\n", meanControl_dim(3));
    
    fprintf("\n-- MEAN TOTAL IMPULSE --\n");
    fprintf("Total Impulse R = %.2f [Ns]\n", totalImpulse(1));
    fprintf("Total Impulse V = %.2f [Ns]\n", totalImpulse(2));
    fprintf("Total Impulse H = %.2f [Ns]\n", totalImpulse(3));
    
    fprintf("\n-- MEAN MANOEUVRE COST --\n");
    fprintf("Total  Mass  Used = %.2f \x00B1 %.2f [kg]\n", meanMass, sigmaMass);
    fprintf("Total DeltaV Used = %.2f \x00B1 %.2f [m/s]\n", meanMass/dt, sigmaMass/dt);

    %% OPTIMAL TRAJECTORY UTILIZATION
    

    %% GNC Execution Time
    fprintf("\n-- MEAN GNC EXECUTION TIME --\n")
    fprintf("Exec Time = %.2g \x00B1 %.2g [ms]\n", meanExecTime*1e3, sigmaExecTime*1e3);

    %% TOF
    fprintf("\n-- MEAN FLIGHT DATA --\n")
    fprintf("MEAN TIME OF FLIGHT   = %.2f \x00B1 %.2f [min]\n", meanTOF*param.tc/60, sigmaTOF*param.tc/60);
    fprintf("MEAN INITIAL DISTANCE = %.2f \x00B1 %.2f [km]\n", meanICp, sigmaICp);
    fprintf("MEAN INITIAL VELOCITY = %.2f \x00B1 %.2f [m/s]\n", meanICv*1e3, sigmaICv*1e3);


    fprintf("\n===========================================================\n")
end