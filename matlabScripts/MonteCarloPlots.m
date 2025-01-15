function [] = MonteCarloPlots(data)
    phaseID = data.phaseID;
    param = data.param;
    n_population = data.n_population;
    timeHistory = data.timeHistory;
    trajectory = data.trajectory;
    fail = data.fail;
    success = data.success;


%%
    failRate = sum(fail)/(n_population)*100;
    fprintf("FAIL RATE: %.2f%%\n",failRate);

    successRate = sum(success)/(n_population)*100;
    fprintf("FAIL RATE: %.2f%%\n",successRate);
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

    fprintf("RENDERING ...\n\n");
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

    Rmax = max(terminalState_conv(:,1,sim_id)*1e2);
    Vmax = max(terminalState_conv(:,2,sim_id)*1e2);
    Hmax  = max(terminalState_conv(:,3,sim_id)*1e2);
    sRmax = terminalState_conv(:,4,sim_id);
    sVmax = terminalState_conv(:,5,sim_id);
    sHmax  = terminalState_conv(:,6,sim_id);

    subplot(2,2,1)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-11, 11])
    ylim([-11, 11])
    xlabel("R-BAR [cm]");
    ylabel("H-BAR [cm]");
    title("position")

    subplot(2,2,2)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-11, 11])
    ylim([-11, 11])
    xlabel("R-BAR [cm]");
    ylabel("V-BAR [cm]");
    title("position")

    subplot(2,2,3)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-.1, .1])
    ylim([-.1, .1])
    xlabel("R-BAR [m/s]");
    ylabel("H-BAR [m/s]");
    title("velocity")

    subplot(2,2,4)
    grid minor
    axis equal
    xline(0,'Color','black')
    yline(0,'Color','black')
    xlim([-.1, .1])
    ylim([-.1, .1])
    xlabel("R-BAR [m/s]");
    ylabel("V-BAR [m/s]");
    title("velocity")

    fprintf("DONE. Press CTRL+C to close the plots...")
    pause();