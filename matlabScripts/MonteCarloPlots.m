function [] = MonteCarloPlots(data,eachplotbool)
    
    if nargin < 2
        eachplotbool = 0;
    elseif eachplotbool == 1
        close all
    end

    phaseID = data.phaseID;
    param = data.param;
    n_population = data.n_population;
    timeHistory = data.timeHistory;
    trajectory = data.trajectory;
    controlAction = data.controlAction;
    OBoTUsage = [zeros(1,n_population); data.OBoTUsage];
    AgentAction = data.AgentAction; % Azione agente: (timestep, n_simulation)
    terminalState = data.terminalState;
    if isfield(data,"RLGNCratio")
        RLGNCratio = data.RLGNCratio;
    else
        RLGNCratio = 100;
    end

    %% print general data information
    [meanFinalState, sigmaFinalState] = MonteCarloInfo(data);

    %% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
    fprintf("PLOTTING ...\n");
    % compute actual relative dynamics for each simulation

    dynamicsHistory = NaN(length(timeHistory),6,length(n_population));

    for sim_id = 1:n_population
        fprintf(" Computing actual relative dynamics: simulation %d OUT OF %d\n",sim_id,n_population)

        %% REMOVE EXTRA VALUES
        soluz = trajectory(:,:,sim_id);
        indicezeri = soluz(:,7:12) == 0;
        soluz(indicezeri,1:6) = 0;
        indiceValori = ~(soluz(:,1) == 0 & soluz(:,2) == 0 & soluz(:,3) == 0);
        soluz = soluz(indiceValori,:);
        time = timeHistory(indiceValori);
        control = controlAction(indiceValori,:,sim_id);

        %% ROTATE REAL DYNAMICS TO LVLH
        relDynami = zeros(length(time),6);
        for id = 1:length(time)
            [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
            relDynami(id,1:6) = rotatedRelativeState;
        end

        relDynami = relDynami.*param.xc;
        relDynami(:,4:6) = relDynami(:,4:6)./param.tc;
        dynamicsHistory(1:length(time),:,sim_id) = relDynami;

        if eachplotbool
            fprintf(" PLOT %d OUT OF %d\n",sim_id,n_population)

            figure(1)
            % plot constraints
            quiver3(0,0,0,1,0,0,'r','LineWidth',1);
            hold on
            quiver3(0,0,0,0,1,0,'r','LineWidth',1);
            quiver3(0,0,0,0,0,1,'r','LineWidth',1);
            plot3(0,0,0,'r*','LineWidth',2)

            if phaseID == 1
                plot3(0,-4,0,'kd','LineWidth',1)
            end

            %% 1 Plot Control Action, Controlled Relative Dynamics, and Velocity
            figure(2)
            % Subplot 1: Control Action
            subplot(3,1,1)
            plot(time * param.tc / 60, control(:,1), 'LineWidth', 1.1); hold on
            plot(time * param.tc / 60, control(:,2), 'LineWidth', 1.1);
            plot(time * param.tc / 60, control(:,3), 'LineWidth', 1.1);
            % Subplot 2: Controlled Relative Dynamics
            subplot(3,1,2)
            plot(time * param.tc / 60, relDynami(:,1), 'LineWidth', 1); hold on
            plot(time * param.tc / 60, relDynami(:,2), 'LineWidth', 1);
            plot(time * param.tc / 60, relDynami(:,3), 'LineWidth', 1);
            % Subplot 3: Controlled Relative Velocity
            subplot(3,1,3)
            plot(time * param.tc / 60, relDynami(:,4).*1e3, 'LineWidth', 1); hold on
            plot(time * param.tc / 60, relDynami(:,5).*1e3, 'LineWidth', 1);
            plot(time * param.tc / 60, relDynami(:,6).*1e3, 'LineWidth', 1);
    
            %% 2
            figure(1)
            plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)
            plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2);
        end
    end
    if eachplotbool
        figure(1)
        if phaseID == 1
            plotConstraintsVisualization(1e3,'S','yellow')
            plotConstraintsVisualization(200,'S')
            plotConstraintsVisualization(2.5e3,'S','black')
        elseif phaseID == 2
            plotConstraintsVisualization(1e3,'C')
        end
        if phaseID == 1
            legend("Target LVLH","","","Target Position","Holding State","Chaser Initial Positions",'Location','best')
        else
            legend("Target LVLH","","","Target Position","Chaser Initial Positions","Relative Trajectory",'Location','best')
        end
        
        axis equal
        xlabel("R-bar [km]")
        ylabel("V-bar [km]")
        zlabel("H-bar [km]")
    
        title("Relative Dynamics")
        grid on
    
        % Plot Control Action, Controlled Relative Dynamics, and Velocity
        figure(2)
        
        % Subplot 1: Control Action
        subplot(3,1,1)
        grid on;
        title("Control Action [LVLH]");
        legend("R-BAR", "V-BAR", "H-BAR", 'Location', 'best');
        xlabel("Time [min]");
        ylabel("Control Action [-]");
        
        % Subplot 2: Controlled Relative Dynamics
        subplot(3,1,2)
        grid on;
        title("Controlled Relative Dynamics [LVLH]");
        legend("R-BAR", "V-BAR", "H-BAR", 'Location', 'best');
        xlabel("Time [min]");
        ylabel("Position [km]");
        
        % Subplot 3: Controlled Relative Velocity
        subplot(3,1,3)
        grid on;
        title("Controlled Relative Velocity [LVLH]");
        legend("R-BAR", "V-BAR", "H-BAR", 'Location', 'best');
        xlabel("Time [min]");
        ylabel("Velocity [m/s]");
    end
 
    %% other plot
    figure()
    % convert to m/s
    terminalState_conv = terminalState.*1e3; % meters
    terminalState_conv(4:6,:) = terminalState_conv(4:6,:);

    % plots (terminal state precision)
    for sim_id = 1:n_population
        subplot(2,2,1)
        plot(terminalState_conv(3,sim_id)*1e2,terminalState_conv(1,sim_id)*1e2,'b.','MarkerSize',8);
        hold on;
        subplot(2,2,2)
        plot(terminalState_conv(2,sim_id)*1e2,terminalState_conv(1,sim_id)*1e2,'b.','MarkerSize',8);
        hold on;
        subplot(2,2,3)
        plot(terminalState_conv(6,sim_id),terminalState_conv(4,sim_id),'b.','MarkerSize',8);
        hold on;
        subplot(2,2,4)
        plot(terminalState_conv(5,sim_id),terminalState_conv(4,sim_id),'b.','MarkerSize',8);
        hold on;
    end

    subplot(2,2,1); 
    grid minor; axis equal; 
    xline(0,'Color','black'); yline(0,'Color','black'); 
    xlim([-11, 11]); ylim([-11, 11]); 
    xlabel("H-BAR [cm]"); ylabel("R-BAR [cm]"); 
    title("position")
    subplot(2,2,2)
    grid minor; axis equal; 
    xline(0,'Color','black'); yline(0,'Color','black'); 
    xlim([-11, 11]); ylim([-11, 11]); 
    xlabel("V-BAR [cm]"); ylabel("R-BAR [cm]");
    title("position")
    subplot(2,2,3)
    grid minor; axis equal; 
    xline(0,'Color','black'); yline(0,'Color','black'); 
    xlim([-.1, .1]); ylim([-.1, .1]);
    xlabel("H-BAR [m/s]"); ylabel("R-BAR [m/s]");
    title("velocity")
    subplot(2,2,4)
    grid minor; axis equal; 
    xline(0,'Color','black'); yline(0,'Color','black'); 
    xlim([-.1, .1]); ylim([-.1, .1]);
    xlabel("V-BAR [m/s]"); ylabel("R-BAR [m/s]");
    title("velocity")

    fprintf("PLOTTING OTHER STUFF...\n")
    %% add a plot on the controlAction
    if isfield(data,"controlAction")

        % Inizializza array per le norme delle distanze
        norm_distances = NaN(n_population,length(timeHistory));
        norm_distances_agent = NaN(n_population,ceil(length(timeHistory)/RLGNCratio));
        agent_actions = NaN(n_population,ceil(length(timeHistory)/RLGNCratio)+1);
        norm_controls = NaN(n_population,length(timeHistory));
        usage_flags = NaN(n_population,length(timeHistory));

        % Calcolo delle norme delle distanze e raccolta delle azioni
        for sim_id = 1:n_population

            % Norma delle distanze (x, y, z)
            distances = vecnorm(dynamicsHistory(:, 1:3, sim_id), 2, 2);
    

            % Azioni dell'agente
            distancesAgent = vecnorm(dynamicsHistory(1:RLGNCratio:end, 1:3, sim_id), 2, 2);
            actions = [0; AgentAction(1:RLGNCratio:end, sim_id)]; % the first action is always (SKIP)

            % Norma della control action
            control_norm = vecnorm(controlAction(:, :, sim_id), 2, 2); % Norma euclidea dei controlli
    
            % Flag di utilizzo della traiettoria ottimale
            usage = OBoTUsage(:, sim_id)';

            % Salva i risultati
            norm_distances(sim_id,:) = distances;
            norm_distances_agent(sim_id,:) = distancesAgent;
            agent_actions(sim_id,:) = actions;
            norm_controls(sim_id,:) = control_norm;
            usage_flags(sim_id,:) = usage;
        end
    
        % Definizione dei bin per aggregare le distanze
        minDist = min(min(norm_distances));
        maxDist = max(max(norm_distances));
        nBins = 50;  % Numero di bin
        binEdges = linspace(minDist, maxDist, nBins+1);
        binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

        %% Media della norma della control action per bin
        average_controls = zeros(1, nBins)./0;
        for b = 1:nBins
            % Trova gli indici delle distanze che appartengono al bin
            inBin = norm_distances >= binEdges(b) & norm_distances < binEdges(b+1);
    
            % Calcola la media della norma della control action per questo bin
            if any(inBin,'all')
                average_controls(b) = mean(norm_controls(inBin));
            else
                average_controls(b) = NaN; % Per bin vuoti
            end
        end

        % Plot della media della norma della control action rispetto alla norma della distanza
        figure;
        plot(binCenters, average_controls, 'b-', 'LineWidth', 1.5);
        grid on;
        xlabel('||\rho|| [km]');
        ylabel('Mean of the norm of the control action [-]');
        title('Mean Control Action');
    end
    
    %% add a plot on the AgentAction

    % Inizializza array per frequenze
    action_distribution = zeros(nBins, 3);  % Per azioni 0, 1, 2
    predominant_action = zeros(1, nBins);  % Azione predominante

    % Calcola la distribuzione delle azioni per ciascun bin
    for b = 1:nBins
        % Trova gli indici delle distanze che appartengono al bin
        inBin = norm_distances_agent >= binEdges(b) & norm_distances_agent < binEdges(b+1);

        if any(inBin,'all')
            % Conta le frequenze delle azioni nel bin
            actions_in_bin = agent_actions(inBin);
            action_distribution(b, 1) = sum(actions_in_bin == 0);  % Azione 0
            action_distribution(b, 2) = sum(actions_in_bin == 1);  % Azione 1
            action_distribution(b, 3) = sum(actions_in_bin == 2);  % Azione 2

            % Azione predominante (moda)
            [~, predominant_action(b)] = max(action_distribution(b, :));
        else
            predominant_action(b) = NaN;  % Bin vuoto
        end
    end

    % Normalizza la distribuzione per rappresentarla in percentuale
    action_distribution = action_distribution ./ sum(action_distribution, 2) * 100;

    % Plot della distribuzione delle azioni
    figure;
    subplot(3,2,[1 3])
    b = bar(binCenters, action_distribution, 'stacked');
    b(1).FaceColor = [0.2 0.6 1.0]; % Blue
    b(2).FaceColor = [1.0 1.0 0.4]; % Yellow
    b(3).FaceColor = [1.0 0.2 0.2]; % Red

    grid on;
    xlabel('norm of relative distance ||\rho|| [km]');
    ylabel('AgentAction distrubution [%]');
    title('Distribuzione delle azioni dell''agente rispetto alla norma della distanza');
    legend('SKIP (0)', 'COMPUTE (1)', 'DELETE (2)', 'Location', 'best');
    ylim([0 100]);
    xlim([0 maxDist]);

    % Plot dell'azione predominante
    subplot(3,2,5)
    plot(binCenters, predominant_action-1, 'bd', 'LineWidth', 1.5);
    grid on;
    xlabel('norm of relative distance ||\rho|| [km]');
    ylabel('predominant AgentAction');
    title('predominant AgentAction with respect to ||\rho||');
    ylim([-0.5, 2.5]);  % Per evidenziare i valori discreti 0, 1, 2
    yticks(0:2);
    xlim([0 maxDist]);
    
    %%
    % Definizione dei bin per aggregare le distanze
    % Calcola la frequenza d'uso della traiettoria ottimale per bin
    optimal_trajectory_usage = zeros(1, nBins)./0;
    for b = 1:nBins
        % Trova gli indici delle distanze che appartengono al bin
        inBin = norm_distances >= binEdges(b) & norm_distances < binEdges(b+1);

        % Calcola la percentuale d'uso della traiettoria ottimale
        if any(inBin,'all')
            optimal_trajectory_usage(b) = mean(usage_flags(inBin)) * 100; % Percentuale
        else
            optimal_trajectory_usage(b) = NaN; % Gestisce bin vuoti
        end
    end

    % Rimuovi eventuali NaN per evitare problemi nel plot
    % Plot della frequenza d'uso della traiettoria ottimale rispetto alla norma della distanza
    % plot(binCenters, optimal_trajectory_usage, 'b-', 'LineWidth', 1.5);
    % grid on;
    % xlabel('||\rho|| [km]');
    % ylabel('Usage of OBoT [%]');
    % title('Frequency of usage of the optimal trajectory');

    %%
    % Inizializza array per le metriche
    distances_vbar = [];   % Distanza lungo V-BAR (velocità, componente y)
    distances_rh = [];     % Distanza sul piano R-H (posizione, componenti x e z)
    usage_flags = [];      % Flag di uso della traiettoria ottimale

    % Calcolo delle metriche per ciascuna simulazione
    for sim_id = 1:n_population
        % Estrarre le componenti della posizione e della velocità
        positions = dynamicsHistory(:, 1:3, sim_id); % Componenti di posizione (x, y, z)

        % Calcolare la distanza sul piano R-H e lungo V-BAR
        distances_rh_sim = sqrt(positions(:, 1).^2 + positions(:, 3).^2); % rho_R^2 + rho_H^2
        distances_vbar_sim = abs(positions(:, 2)); % Norma lungo V-BAR

        % Flag di uso della traiettoria ottimale
        usage_sim = OBoTUsage(:, sim_id);

        % Accumula i dati
        distances_rh = [distances_rh; distances_rh_sim];
        distances_vbar = [distances_vbar; distances_vbar_sim];
        usage_flags = [usage_flags; usage_sim];
    end

    % Definizione dei bin per aggregare i dati
    nBins = 30;  % Numero di bin
    binEdges_rh = linspace(min(distances_rh), max(distances_rh), nBins+1);
    binEdges_vbar = linspace(min(distances_vbar), max(distances_vbar), nBins+1);

    % Centri dei bin
    binCenters_rh = (binEdges_rh(1:end-1) + binEdges_rh(2:end)) / 2;
    binCenters_vbar = (binEdges_vbar(1:end-1) + binEdges_vbar(2:end)) / 2;

    usage_frequency = NaN(nBins);

    % Calcolo della frequenza di utilizzo per ciascun bin
    for i = 1:nBins
        for j = 1:nBins
            % Trova gli indici dei dati che appartengono al bin corrente
            inBin = distances_rh >= binEdges_rh(i) & distances_rh < binEdges_rh(i+1) & ...
                    distances_vbar >= binEdges_vbar(j) & distances_vbar < binEdges_vbar(j+1);
    
            % Calcola la frequenza di utilizzo della traiettoria ottimale
            if any(inBin)
                usage_frequency(j, i) = mean(usage_flags(inBin)) * 100; % Percentuale
            else
                usage_frequency(j, i) = NaN; % Gestione di bin vuoti
            end
        end
    end
    
    % Plot 3D con bar3
    subplot(3,2,[2 4 6]);
    heatmap(binCenters_rh, binCenters_vbar, usage_frequency); % Istogramma 3D
    
    % Personalizzazione del plot
    colormap(parula);
    colorbar;
    xlabel('R-H-BAR distance sqrt(\rho_R^2 + \rho_H^2) [km]');
    ylabel('V-BAR distance ||\rho_V|| [km]');
    title('Usage of Optimal Trajectory [%]');
    grid on;
    
    %%  for python use
    fprintf("RENDERING ...\n\n");
    %fprintf("DONE. Press CTRL+C to close the plots...")
    %pause();

end
