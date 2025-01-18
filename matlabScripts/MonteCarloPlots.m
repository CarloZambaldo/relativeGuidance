function [] = MonteCarloPlots(data)
close all

    phaseID = data.phaseID;
    param = data.param;
    n_population = data.n_population;
    timeHistory = data.timeHistory;
    trajectory = data.trajectory;
    controlAction = data.controlAction;
    OBoTUsage = data.OBoTUsage;
    AgentAction = data.AgentAction; % Azione agente: (timestep, n_simulation)
    fail = data.fail;
    success = data.success;

%%
    failRate = sum(fail)/(n_population)*100;
    fprintf("FAIL RATE: %.2f%%\n",failRate);

    successRate = sum(success)/(n_population)*100;
    fprintf("SUCCESS RATE: %.2f%%\n",successRate);

    %% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
    fprintf("PLOTTING ...\n");
    % compute actual relative dynamics for each simulation

    figure()
    % plot constraints
    quiver3(0,0,0,1,0,0,'r','LineWidth',1);
    hold on
    quiver3(0,0,0,0,1,0,'r','LineWidth',1);
    quiver3(0,0,0,0,0,1,'r','LineWidth',1);
    plot3(0,0,0,'r*','LineWidth',2)

    terminalState = zeros(1,6,length(n_population));
    for sim_id = 1:n_population
        fprintf(" PLOT %d OUT OF %d\n",sim_id,n_population)
        soluz = trajectory(:,:,sim_id);

        indicezeri = soluz(:,7:12) == 0;
        soluz(indicezeri,1:6) = 0;

        indiceValori = ~(soluz(:,1) == 0 & soluz(:,2) == 0 & soluz(:,3) == 0);
        soluz = soluz(indiceValori,:);
        time = timeHistory(indiceValori);

        relDynami = zeros(length(time),6);

        for id = 1:length(time)
            [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
            relDynami(id,1:6) = rotatedRelativeState;
        end

        terminalState(:,:,sim_id) = relDynami(end,1:6);

        relDynami = relDynami.*param.xc;
        plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)
        plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'LineWidth',1.2);
    end

    if phaseID == 1
        plotConstraintsVisualization(1e3,'S','yellow')
        plotConstraintsVisualization(200,'S')
        plotConstraintsVisualization(2.5e3,'S','Color','black')
    elseif phaseID == 2
        plotConstraintsVisualization(1e3,'C')
    end
    legend("Target LVLH","","","","","Initial Positions",'Location','best')
    axis equal
    xlabel("R-bar [km]")
    ylabel("V-bar [km]")
    zlabel("H-bar [km]")

    title("Relative Dynamics")
    grid on


    %% second plot
    figure()
    % convert to m/s
    terminalState_conv = terminalState.*param.xc*1e3;
    terminalState_conv(:,4:6,:) = terminalState_conv(:,4:6,:)./param.tc;

    % plots
    for sim_id = 1:n_population
        subplot(2,2,1)
        plot(terminalState_conv(:,3,sim_id)*1e2,terminalState_conv(:,1,sim_id)*1e2,'b.','MarkerSize',8);
        hold on;

        subplot(2,2,2)
        plot(terminalState_conv(:,2,sim_id)*1e2,terminalState_conv(:,1,sim_id)*1e2,'b.','MarkerSize',8);
        hold on;

        subplot(2,2,3)
        plot(terminalState_conv(:,6,sim_id),terminalState_conv(:,4,sim_id),'b.','MarkerSize',8);
        hold on;

        subplot(2,2,4)
        plot(terminalState_conv(:,5,sim_id),terminalState_conv(:,4,sim_id),'b.','MarkerSize',8);
            hold on;

    end

    subplot(2,2,1)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-11, 11])
    ylim([-11, 11])
    xlabel("H-BAR [cm]");
    ylabel("R-BAR [cm]");
    title("position")

    subplot(2,2,2)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-11, 11])
    ylim([-11, 11])
    xlabel("V-BAR [cm]");
    ylabel("R-BAR [cm]");
    title("position")

    subplot(2,2,3)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-.1, .1])
    ylim([-.1, .1])
    xlabel("H-BAR [m/s]");
    ylabel("R-BAR [m/s]");
    title("velocity")

    subplot(2,2,4)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-.1, .1])
    ylim([-.1, .1])
    xlabel("V-BAR [m/s]");
    ylabel("R-BAR [m/s]");
    title("velocity")



    fprintf("PLOTTING OTHER STUFF...")
    %% add a plot on the controlAction
    if isfield(data,"controlAction")
        % Supponiamo che data sia una struttura con i campi trajectory e controlAction.
        % Ecco come creare le variabili e calcolare la norma della control action media rispetto alla norma della posizione.
        
        % Inizializza array per le norme
        norm_distances = [];
        norm_controls = [];
    
        % Calcolo delle norme per ogni simulazione
        for sim_id = 1:n_population
            % Norma delle distanze (x, y, z)
            positions = trajectory(:, 1:3, sim_id); % Prendi solo x, y, z
            distances = vecnorm(positions, 2, 2);   % Norma euclidea r = sqrt(x^2 + y^2 + z^2)
    
            % Norma della control action
            controls = controlAction(:, :, sim_id); % Prendi i controlli
            control_norm = vecnorm(controls, 2, 2); % Norma euclidea dei controlli
    
            % Salva i risultati
            norm_distances = [norm_distances; distances];
            norm_controls = [norm_controls; control_norm];
        end
    
        % Definizione dei bin per aggregare le distanze
        minDist = min(norm_distances);
        maxDist = max(norm_distances);
        nBins = 50;  % Numero di bin
        binEdges = linspace(minDist, maxDist, nBins+1);
        binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
    
        % Media della norma della control action per bin
        average_controls = zeros(1, nBins);
        for b = 1:nBins
            % Trova gli indici delle distanze che appartengono al bin
            inBin = norm_distances >= binEdges(b) & norm_distances < binEdges(b+1);
    
            % Calcola la media della norma della control action per questo bin
            if any(inBin)
                average_controls(b) = mean(norm_controls(inBin));
            else
                average_controls(b) = NaN; % Per bin vuoti
            end
        end
    
        % Rimuovi eventuali NaN per evitare problemi nel plot
        validIndices = ~isnan(average_controls);
        binCenters = binCenters(validIndices);
        average_controls = average_controls(validIndices);
    
        % Plot della media della norma della control action rispetto alla norma della distanza
        figure;
        plot(binCenters*param.xc, average_controls, 'b-', 'LineWidth', 1.5);
        grid on;
        xlabel('||\rho|| [km]');
        ylabel('Mean of the norm of the control action [-]');
        title('Mean Control Action');
    end
    
    %% add a plot on the AgentAction
    % Inizializza array per le norme delle distanze
    norm_distances = [];
    agent_actions = [];

    % Calcolo delle norme delle distanze e raccolta delle azioni
    for sim_id = 1:n_population
        % Norma delle distanze (x, y, z)
        positions = trajectory(:, 1:3, sim_id);
        distances = vecnorm(positions, 2, 2);

        % Azioni dell'agente
        actions = AgentAction(:, sim_id);

        % Salva i risultati
        norm_distances = [norm_distances; distances];
        agent_actions = [agent_actions; actions];
    end

    % Definizione dei bin per aggregare le distanze
    minDist = min(norm_distances);
    maxDist = max(norm_distances);
    nBins = 50;  % Numero di bin
    binEdges = linspace(minDist, maxDist, nBins+1);
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

    % Inizializza array per frequenze
    action_distribution = zeros(nBins, 3);  % Per azioni 0, 1, 2
    predominant_action = zeros(1, nBins);  % Azione predominante

    % Calcola la distribuzione delle azioni per ciascun bin
    for b = 1:nBins
        % Trova gli indici delle distanze che appartengono al bin
        inBin = norm_distances >= binEdges(b) & norm_distances < binEdges(b+1);

        if any(inBin)
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
    subplot(2,2,[1 3])
    b = bar(binCenters * param.xc, action_distribution, 'stacked');
    b(1).FaceColor = [0.2 0.6 1.0]; % Blue
    b(2).FaceColor = [1.0 1.0 0.4]; % Yellow
    b(3).FaceColor = [1.0 0.2 0.2]; % Red

    grid on;
    xlabel('norm of relative distance ||\rho|| [km]');
    ylabel('AgentAction distrubution [%]');
    title('Distribuzione delle azioni dell''agente rispetto alla norma della distanza');
    legend('SKIP (0)', 'COMPUTE (1)', 'DELETE (2)', 'Location', 'best');

    % Plot dell'azione predominante
    subplot(2,2,2)
    plot(binCenters*param.xc, predominant_action, 'b.-', 'LineWidth', 1.5);
    grid on;
    xlabel('norm of relative distance ||\rho|| [km]');
    ylabel('predominant AgentAction');
    title('predominant AgentAction with respect to ||\rho||');
    ylim([-0.5, 2.5]);  % Per evidenziare i valori discreti 0, 1, 2
    yticks(0:2);
    
    %%
    % Calcola la norma della posizione
    norm_distances = [];
    usage_flags = [];

    for sim_id = 1:n_population
        % Norma della distanza per la simulazione corrente
        positions = trajectory(:, 1:3, sim_id); % Prendi solo x, y, z
        distances = vecnorm(positions, 2, 2);   % Norma euclidea

        % Flag di utilizzo della traiettoria ottimale
        usage = OBoTUsage(:, sim_id);

        % Salva i dati
        norm_distances = [norm_distances; distances];
        usage_flags = [usage_flags; usage];
    end

    % Definizione dei bin per aggregare le distanze
    minDist = min(norm_distances);
    maxDist = max(norm_distances);
    nBins = 50;  % Numero di bin
    binEdges = linspace(minDist, maxDist, nBins+1);
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

    % Calcola la frequenza d'uso della traiettoria ottimale per bin
    optimal_trajectory_usage = zeros(1, nBins);
    for b = 1:nBins
        % Trova gli indici delle distanze che appartengono al bin
        inBin = norm_distances >= binEdges(b) & norm_distances < binEdges(b+1);

        % Calcola la percentuale d'uso della traiettoria ottimale
        if any(inBin)
            optimal_trajectory_usage(b) = mean(usage_flags(inBin)) * 100; % Percentuale
        else
            optimal_trajectory_usage(b) = NaN; % Gestisce bin vuoti
        end
    end

    % Rimuovi eventuali NaN per evitare problemi nel plot
    validIndices = ~isnan(optimal_trajectory_usage);
    binCenters = binCenters(validIndices);
    optimal_trajectory_usage = optimal_trajectory_usage(validIndices);

    % Plot della frequenza d'uso della traiettoria ottimale rispetto alla norma della distanza
    subplot(2,2,4);
    plot(binCenters*param.xc, optimal_trajectory_usage, 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('||\rho|| [km]');
    ylabel('Usage of computed optimal Trajectory [%]');
    title('Frequency of usage of the optimal trajectory');

    %%
    % Inizializza array per le metriche
    distances_vbar = [];   % Distanza lungo V-BAR (velocità, componente y)
    distances_rh = [];     % Distanza sul piano R-H (posizione, componenti x e z)
    usage_flags = [];      % Flag di uso della traiettoria ottimale

    % Calcolo delle metriche per ciascuna simulazione
    for sim_id = 1:n_population
        % Estrarre le componenti della posizione e della velocità
        positions = trajectory(:, 1:3, sim_id); % Componenti di posizione (x, y, z)
        velocities = trajectory(:, 4:6, sim_id); % Componenti di velocità (vx, vy, vz)

        % Calcolare la distanza sul piano R-H e lungo V-BAR
        distances_rh_sim = sqrt(positions(:, 1).^2 + positions(:, 3).^2); % rho_R^2 + rho_H^2
        distances_vbar_sim = abs(velocities(:, 2)); % Norma lungo V-BAR (componente vy)

        % Flag di uso della traiettoria ottimale
        usage_sim = OBoTUsage(:, sim_id);

        % Accumula i dati
        distances_rh = [distances_rh; distances_rh_sim];
        distances_vbar = [distances_vbar; distances_vbar_sim];
        usage_flags = [usage_flags; usage_sim];
    end

    % Definizione dei bin per aggregare i dati
    nBins = 50;  % Numero di bin
    binEdges_rh = linspace(min(distances_rh), max(distances_rh), nBins+1);
    binEdges_vbar = linspace(min(distances_vbar), max(distances_vbar), nBins+1);

    % Inizializza matrice per la frequenza d'uso
    usage_frequency = zeros(nBins, nBins);
    % Centri dei bin
    binCenters_rh = (binEdges_rh(1:end-1) + binEdges_rh(2:end)) / 2;
    binCenters_vbar = (binEdges_vbar(1:end-1) + binEdges_vbar(2:end)) / 2;
    
    % Creazione della griglia per il plot
    [X, Y] = meshgrid(binCenters_rh, binCenters_vbar);
    
    % Rimpiazza i NaN con 0 per evitare problemi con il plot
    usage_frequency(isnan(usage_frequency)) = 0;
    
    % Plot 3D con bar3
    figure;
    hold on;
    for i = 1:size(usage_frequency, 2) % Ciclo sui bin lungo R-H
        h = bar3(binCenters_vbar, usage_frequency(:, i)); % Istogramma 3D
        % Imposta la larghezza delle barre e il colore
        % for k = 1:length(h)
        %     h(k).FaceColor = 'flat'; % Consenti colori personalizzati
        %     h(k).CData = repmat(i, size(h(k).ZData, 1), 1); % Gradient color per bin R-H
        % end
    end
    hold off;
    
    % Personalizzazione del plot
    colormap(jet);
    colorbar;
    xlabel('R-H-BAR distance sqrt(\rho_R^2 + \rho_H^2)');
    ylabel('V-BAR distance ||\rho_V||');
    zlabel('Usage of Optimal Trajectory [%]');
    title('Optimal Trajectory Usage Frequency (3D Histogram)');
    grid on;

    
    %%  for python use
    fprintf("RENDERING ...\n\n");
    %fprintf("DONE. Press CTRL+C to close the plots...")
    %pause();

end
