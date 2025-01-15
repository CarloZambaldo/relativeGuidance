function [] = matlabPLOT(data, constraintPlotBool)
    %% extract data from the input
    %      "phaseID"
    %      "timeHistory"
    %      "fullStateHistory" 
    %      "controlActionHistory_L" 
    %      "AgentActionHistory" 
    %      "constraintViolationHistory" 
    %      "terminationCause"

    if nargin < 2
        constraintPlotBool = 0;
    end

    phaseID = data.phaseID;
    param = data.param;
    time = data.timeHistory;
    soluz = data.fullStateHistory();

    relDynami = zeros(length(time),6);

    for id = 1:length(time)
        [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
        relDynami(id,1:6) = rotatedRelativeState;
    end

    terminalState = relDynami(end,1:6);

    relDynami = relDynami.*param.xc;
    plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1);
    hold on;
    plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'LineWidth',1.2);

    % convert to m/s
    terminalState_conv = terminalState.*param.xc*1e3;
    terminalState_conv(:,4:6) = terminalState_conv(:,4:6)./param.tc;

    %% PLOTTING THE CONE AND THE TRAJECTORIES
    figure(1);


    if constraintPlotBool == 1
        % if 1 plot the constraints
        quiver3(0,0,0,1,0,0,'r','LineWidth',1);
        hold on
        quiver3(0,0,0,0,1,0,'r','LineWidth',1);
        quiver3(0,0,0,0,0,1,'r','LineWidth',1);
        plot3(0,0,0,'r*','LineWidth',2)

        if phaseID == 1
            plotConstraintsVisualization(1e3,'S','yellow')
            plotConstraintsVisualization(200,'S')
            plotConstraintsVisualization(2.5e3,'S','Color','black')
        elseif phaseID == 2
            plotConstraintsVisualization(1e3,'C')
        end
        legend("Initial Positions","Relative Dynamics","LVLH Frame",'Location','best')
        
        axis equal
        xlabel("R-bar [km]")
        ylabel("V-bar [km]")
        zlabel("H-bar [km]")

        title("Relative Dynamics")
        grid on
    end


    %% PLOTTING THE PRECISION ALONG THE AXES
    figure(2);
    subplot(2,2,1)
    plot(terminalState_conv(:,3)*1e2,terminalState_conv(:,1)*1e2,'b.','MarkerSize',8);
    hold on;

    subplot(2,2,2)
    plot(terminalState_conv(:,2)*1e2,terminalState_conv(:,1)*1e2,'b.','MarkerSize',8);
    hold on;

    subplot(2,2,3)
    plot(terminalState_conv(:,6),terminalState_conv(:,4),'b.','MarkerSize',8);
    hold on;

    subplot(2,2,4)
    plot(terminalState_conv(:,5),terminalState_conv(:,4),'b.','MarkerSize',8);
        hold on;


    if plotConstraintsVisualization == 1
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
    end

    %% plotting the agent actions

    %% pause to avoid closing the plots while running in pythons
    pause();
end

