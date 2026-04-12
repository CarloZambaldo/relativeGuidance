function [meanFinalState,sigmaFinalState, MC_results] = MonteCarloInfo(data)

    phaseID = data.phaseID;
    param = data.param;
    n_population = double(data.n_population);
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
    execTime = data.CPUExecTimeHistory;
    
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
    fprintf("Noise level: %s\n",data.param.noisePercent)
    fprintf("Agent model used: %s\n", agentModelName);
    fprintf("Seed used: %s\n", string(data.seed));
    fprintf("\n")
    
    %%
    fprintf("-- DATA STATISTICS --\n")
    successRate = sum(success)/double(n_population)*100;
    fprintf("SUCCESS RATE:  %3.2f %%\n",successRate);

    failRate = sum(fail)/double(n_population)*100;
    fprintf("FAIL RATE:     %3.2f %%\n",failRate);

    notConverged = 100 - failRate - successRate;
    fprintf("NOT CONVERGED: %3.2f %%\n",notConverged);

    no_not_conv = length(data.fail) - sum(success) - sum(fail);
    if no_not_conv > 0
        indici = find(data.success==0 & data.fail==0);
        fprintf("  Index of not converged: ");
        for i = 1:no_not_conv
            fprintf("%d, ", indici(i));
        end
    end
    
    fprintf("\n")


    %% COMPUTE STUFF
    MC_results.meanControl_dim = zeros(1,3); % THIS IS IN NEWTON!!
    MC_results.totalImpulse = zeros(1,3);
    totalMassUsed = 0;
    MC_results.obotusagetru = 0;
    simulationMass = zeros(n_population,1);
    
    for sim_id = 1:n_population
        MC_results.obotusagetru = MC_results.obotusagetru + sum(OBoTUsage(1:endTimeIx(sim_id),sim_id)) / endTimeIx(sim_id);
        time_dim = timeHistory(1:endTimeIx(sim_id)) .* param.tc;
        dt = 1/param.freqGNC .* param.tc;

        control_dim = double(param.chaserMass) .* controlAction(1:endTimeIx(sim_id),:,sim_id) .* param.xc * 1e3 ./ param.tc^2;
        
        MC_results.meanControl_dim = MC_results.meanControl_dim + trapz(time_dim,control_dim,1);

        MC_results.totalImpulse = MC_results.totalImpulse + sum(control_dim(1:end-1,:) .* dt, 1);
        simulationMass(sim_id) = sum(vecnorm(control_dim(1:end-1,:), 2, 1) .* dt) / double(param.chaserSpecificImpulse) / 9.81;
    end

    [MC_results.sigmaICp, MC_results.meanICp] = std(vecnorm(trueRelativeStateHistory_L(1,1:3,:),2,2)*param.xc);
    [MC_results.sigmaICv, MC_results.meanICv] = std(vecnorm(trueRelativeStateHistory_L(1,4:6,:),2,2)*param.xc/param.tc);
    [MC_results.sigmaTOF, MC_results.meanTOF] = std(timeHistory(endTimeIx));
    [MC_results.sigmaMass, MC_results.meanMass] = std(simulationMass);
    [MC_results.sigmaExecTime, MC_results.meanExecTime] = std(execTime(execTime~=0));
    MC_results.meanControl_dim = MC_results.meanControl_dim / double(n_population);
    MC_results.totalImpulse = MC_results.totalImpulse / double(n_population);
    

    MC_results.recomputationsBool = sum(AgentAction == 1, 1);

    MC_results.obotusagetru = MC_results.obotusagetru/n_population;


    %% STATISTICS ON THE FINAL POSITION
    [MC_results.sigmaFinalState, MC_results.meanFinalState] = std(terminalState_dimensional,0,2);
    fprintf("Final Position R-BAR = %.2f \x00B1 %.2f [cm]\n",  MC_results.meanFinalState(1)*1e2, MC_results.sigmaFinalState(1)*1e2);
    fprintf("Final Position V-BAR = %.2f \x00B1 %.2f [cm]\n",  MC_results.meanFinalState(2)*1e2, MC_results.sigmaFinalState(2)*1e2);
    fprintf("Final Position H-BAR = %.2f \x00B1 %.2f [cm]\n",  MC_results.meanFinalState(3)*1e2, MC_results.sigmaFinalState(3)*1e2);
    fprintf("Final Velocity R-BAR = %.2f \x00B1 %.2f [cm/s]\n",MC_results.meanFinalState(4)*1e2, MC_results.sigmaFinalState(4)*1e2);
    fprintf("Final Velocity V-BAR = %.2f \x00B1 %.2f [cm/s]\n",MC_results.meanFinalState(5)*1e2, MC_results.sigmaFinalState(5)*1e2);
    fprintf("Final Velocity H-BAR = %.2f \x00B1 %.2f [cm/s]\n",MC_results.meanFinalState(6)*1e2, MC_results.sigmaFinalState(6)*1e2);
    
    meanFinalState = MC_results.meanFinalState;
    sigmaFinalState = MC_results.sigmaFinalState;

    %% MEAN THRUST REQUIRED
    fprintf("\n-- MEAN CONTROL ACTION --\n");
    fprintf("Mean Control Action R = %.2f [N]\n", MC_results.meanControl_dim(1));
    fprintf("Mean Control Action V = %.2f [N]\n", MC_results.meanControl_dim(2));
    fprintf("Mean Control Action H = %.2f [N]\n", MC_results.meanControl_dim(3));
    
    fprintf("\n-- MEAN TOTAL IMPULSE --\n");
    fprintf("Total Impulse R = %.2f [Ns]\n", MC_results.totalImpulse(1));
    fprintf("Total Impulse V = %.2f [Ns]\n", MC_results.totalImpulse(2));
    fprintf("Total Impulse H = %.2f [Ns]\n", MC_results.totalImpulse(3));
    
    fprintf("\n-- MEAN MANOEUVRE COST --\n");
    fprintf("Total  Mass  Used = %.2f \x00B1 %.2f [kg]\n", MC_results.meanMass, MC_results.sigmaMass);
    fprintf("Total DeltaV Used = %.2f \x00B1 %.2f [m/s]\n", MC_results.meanMass/dt, MC_results.sigmaMass/dt);

    %% OPTIMAL TRAJECTORY UTILIZATION
    

    %% GNC Execution Time
    fprintf("\n-- MEAN GNC EXECUTION TIME --\n")
    fprintf("Exec Time = %.3f \x00B1 %.3f [ms]\n", MC_results.meanExecTime*1e3, MC_results.sigmaExecTime*1e3);
    conversion = 2.65e3/100*24/2; % if computed from the server
    fprintf("Equivalent Exec Time = %.2f \x00B1 %.2f [ms]\n", MC_results.meanExecTime*1e3*conversion, MC_results.sigmaExecTime*1e3*conversion);
    fprintf("Avarage number of recomputations per episode: %.2f\n", mean(MC_results.recomputationsBool));
    fprintf("Avarage use of OBoT per episode: %.2f%%\n", MC_results.obotusagetru*100);
    %figure();semilogy(data.CPUExecTimeHistory(:,1)*1e3*conversion,'LineWidth',0.9); grid minor;

    %% TOF
    fprintf("\n-- MEAN FLIGHT DATA --\n")
    fprintf("MEAN TIME OF FLIGHT   = %.2f \x00B1 %.2f [min]\n", MC_results.meanTOF*param.tc/60, MC_results.sigmaTOF*param.tc/60);
    fprintf("MEAN INITIAL DISTANCE = %.2f \x00B1 %.2f [km]\n", MC_results.meanICp, MC_results.sigmaICp);
    fprintf("MEAN INITIAL VELOCITY = %.2f \x00B1 %.2f [m/s]\n", MC_results.meanICv*1e3, MC_results.sigmaICv*1e3);

    fprintf("\n===========================================================\n")
end