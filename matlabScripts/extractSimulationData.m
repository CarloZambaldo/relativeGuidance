function dataExtracted = extractSimulationData(data, simulationID)
    % This function extracts the data of one simulation from the whole data structure
    % Input:
    %   - data: structure containing multiple simulations' data
    %   - simulationID: ID of the simulation to extract (integer)
    % Output:
    %   - dataExtracted: structure containing data for the specified simulation
    
    % Validate simulationID
    for i = 1:length(simulationID)
        if simulationID(i) < 1 || simulationID(i) > data.n_population
            error('Invalid simulationID. It must be between 1 and %d.', data.n_population);
        end
    end

    % Extract data for the specified simulation
    dataExtracted = data;
    dataExtracted.n_population = length(simulationID);
    dataExtracted.phaseID = data.phaseID;
    dataExtracted.param = data.param;
    dataExtracted.timeHistory = data.timeHistory;  % Same for all simulations
    dataExtracted.trajectory = data.trajectory(:, :, simulationID);
    dataExtracted.AgentAction = data.AgentAction(:, simulationID);
    dataExtracted.controlAction = data.controlAction(:, :, simulationID);
    dataExtracted.constraintViolation = data.constraintViolation(:, simulationID);
    dataExtracted.terminalState = data.terminalState(:, simulationID);
    dataExtracted.fail = data.fail(simulationID);
    dataExtracted.success = data.success(simulationID);
    dataExtracted.OBoTUsage = data.OBoTUsage(:, simulationID);
end
