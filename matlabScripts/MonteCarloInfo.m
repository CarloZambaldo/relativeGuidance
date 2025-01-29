function [meanFinalState,sigmaFinalState] = MonteCarloInfo(data)

    phaseID = data.phaseID;
    param = data.param;
    n_population = data.n_population;
    timeHistory = data.timeHistory;
    trajectory = data.trajectory;
    controlAction = data.controlAction;
    OBoTUsage = [zeros(1,n_population); data.OBoTUsage];
    AgentAction = data.AgentAction; % Azione agente: (timestep, n_simulation)
    fail = data.fail;
    success = data.success;
    terminalState = data.terminalState;
    
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
    fprintf("data contains: %d simulations\n", n_population);
    fprintf("Name of Agent model used: %s\n", agentModelName);

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
    %% STATISTICS ON THE FINAL POSITION
    [sigmaFinalState, meanFinalState] = std(terminalState_dimensional,0,2);
    fprintf("Final Position R-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(1)*1e2, sigmaFinalState(1)*1e2);
    fprintf("Final Position V-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(2)*1e2, sigmaFinalState(2)*1e2);
    fprintf("Final Position H-BAR = %.2f \x00B1 %.2f [cm]\n",  meanFinalState(3)*1e2, sigmaFinalState(3)*1e2);
    fprintf("Final Velocity R-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(4)*1e2, sigmaFinalState(4)*1e2);
    fprintf("Final Velocity V-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(5)*1e2, sigmaFinalState(5)*1e2);
    fprintf("Final Velocity H-BAR = %.2f \x00B1 %.2f [cm/s]\n",meanFinalState(6)*1e2, sigmaFinalState(6)*1e2);
    

    %% MEAN THRUST REQUIRED
    meanControlAction = zeros(3,1);
    totalImpulse = zeros(3,1);
    totalMassUsed = 0;
    
    for sim_id = 1:n_population
        soluz = trajectory(:,:,sim_id);
        indicezeri = soluz(:,7:12) == 0;
        soluz(indicezeri,1:6) = 0;
        indiceValori = ~(soluz(:,1) == 0 & soluz(:,2) == 0 & soluz(:,3) == 0);
        soluz = soluz(indiceValori,:);
        time = timeHistory(indiceValori);
        control = controlAction(indiceValori,:,sim_id);
        
        meanControlAction = meanControlAction + mean(control, 1)';
        dt = param.freqGNC;
        totalImpulse = totalImpulse + sum(control(1:end-1,:) .* dt, 1)';
        totalMassUsed = totalMassUsed + sum(vecnorm(control(1:end-1,:), 2, 2) .* dt) / param.Isp / param.g0;
    end
    
    meanControlAction = meanControlAction / n_population;
    totalImpulse = totalImpulse / n_population;
    totalMassUsed = totalMassUsed / n_population;
    
    fprintf("\n-- MEAN CONTROL ACTION --\n");
    fprintf("Mean Control Action X = %.2f [N]\n", meanControlAction(1));
    fprintf("Mean Control Action Y = %.2f [N]\n", meanControlAction(2));
    fprintf("Mean Control Action Z = %.2f [N]\n", meanControlAction(3));
    
    fprintf("\n-- TOTAL IMPULSE --\n");
    fprintf("Total Impulse X = %.2f [Ns]\n", totalImpulse(1));
    fprintf("Total Impulse Y = %.2f [Ns]\n", totalImpulse(2));
    fprintf("Total Impulse Z = %.2f [Ns]\n", totalImpulse(3));
    
    fprintf("\n-- TOTAL MASS USED --\n");
    fprintf("Total Mass Used = %.4f [kg]\n", totalMassUsed);



    %% OPTIMAL TRAJECTORY UTILIZATION



    fprintf("\n===========================================================\n")
end