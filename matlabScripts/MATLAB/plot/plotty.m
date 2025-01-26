function [] = plotty(adimensionalSolution,phaseID,param)
    global OBoptimalTrajectory

    initialStateTarget_S = adimensionalSolution.y(1,1:6)';
    initialStateChaser_S = adimensionalSolution.y(1,7:12)';
    DeltaIC_S = initialStateChaser_S-initialStateTarget_S;

    %% PLOTTING RELATIVE DYNAMICS INSIDE LVLH FRAME
    sol = adimensionalSolution.y';
    chaserPosition   = sol(:,7:9);
    targetPosition   = sol(:,1:3);
    xc = param.xc;
    
    relDynami = zeros(length(adimensionalSolution.x),6);
    soluz = adimensionalSolution.y';
    for id = 1:length(adimensionalSolution.x) 
        [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
        relDynami(id,1:6) = rotatedRelativeState;
    end
    
    controlAction = adimensionalSolution.controlAction;
    relDynami = relDynami.*xc;
    figure
    if ~isempty(OBoptimalTrajectory) && isfield(OBoptimalTrajectory,"x")
        OBOT = OBoptimalTrajectory.x'.*xc;
        plot3(OBOT(:,1),OBOT(:,2),OBOT(:,3),'m--','LineWidth',1.2);
        hold on
    end
    plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2); hold on
    plot3(0,0,0,'r*','LineWidth',2)
    DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
    DeltaICm = DeltaIC_L(1:3)*xc;
    quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
    plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)
    
    quiver3(relDynami(:,1),relDynami(:,2),relDynami(:,3),controlAction(1,:)',controlAction(2,:)',controlAction(3,:)','g','LineWidth',0.8)
    
    
    quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
    quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
    quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)
    
    holdState = param.holdingState*param.xc; % [km]
    plot3(holdState(1),holdState(2),holdState(3),'db','LineWidth',1);
    
    if phaseID == 1
        plotConstraintsVisualization(1e3,'S','yellow')
        plotConstraintsVisualization(200,'S')
        plotConstraintsVisualization(2.5e3,'S','#808080')
    elseif phaseID == 2
        plotConstraintsVisualization(norm(DeltaIC_S)*param.xc*1e3,'C')
        % plotConstraintsVisualization(200,'C')
    end
    
    legend("Optimal Trajectory","Actual Trajectory","","Initial Condition","","Control Action","Target LVLH","","","Hold Position",'Location','best')
    axis equal
    xlabel("R-bar [km]")
    ylabel("V-bar [km]")
    zlabel("H-bar [km]")
    
    title("Relative Dynamics [CONTROLLED]")
    grid on
    % 
    % xlim([-0.1,0.1])
    % ylim([-0.1,0])
    % zlim([-0.1,0.1])
    

    %% zoom in
    if phaseID == 2
        figure
        if ~isempty(OBoptimalTrajectory) && isfield(OBoptimalTrajectory,"x")
            OBOT = OBoptimalTrajectory.x'.*xc;
            plot3(OBOT(:,1),OBOT(:,2),OBOT(:,3),'m--','LineWidth',1.2);
            hold on
        end
        plot3(relDynami(:,1),relDynami(:,2),relDynami(:,3),'b','LineWidth',1.2); hold on
        plot3(0,0,0,'r*','LineWidth',2)
        DeltaIC_L = convert_S_to_LVLH(initialStateTarget_S,DeltaIC_S,param);
        DeltaICm = DeltaIC_L(1:3)*xc;
        quiver3(0,0,0,relDynami(1,1),relDynami(1,2),relDynami(1,3),'k','LineWidth',.9)
        plot3(relDynami(1,1),relDynami(1,2),relDynami(1,3),'ok','LineWidth',1)
        
        quiver3(relDynami(:,1),relDynami(:,2),relDynami(:,3),controlAction(1,:)',controlAction(2,:)',controlAction(3,:)','g','LineWidth',0.8)
        
        
        quiver3(0,0,0,norm(DeltaICm),0,0,'r','LineWidth',1)
        quiver3(0,0,0,0,norm(DeltaICm),0,'r','LineWidth',1)
        quiver3(0,0,0,0,0,norm(DeltaICm),'r','LineWidth',1)
        
        holdState = param.holdingState*param.xc; % [km]
        plot3(holdState(1),holdState(2),holdState(3),'db','LineWidth',1);
        
        plotConstraintsVisualization(200,'C')

        legend("Optimal Trajectory","Actual Trajectory","","Initial Condition","","Control Action","Target LVLH","","","Hold Position",'Location','best')
        axis equal
        xlabel("R-bar [km]")
        ylabel("V-bar [km]")
        zlabel("H-bar [km]")
        
        title("Relative Dynamics [CONTROLLED]")
        grid on
    
        xlim([-1e-4,1e-4])
        ylim([-1e-4,1e-4])
        zlim([-1e-4,1e-4])
    end


    %%
    adimensionalSolution.t = adimensionalSolution.x;
    
    figure
    subplot(3,1,1)
    plot(adimensionalSolution.t*param.tc/60, controlAction,'LineWidth',1.1)
        
    % stairs(adimensionalSolution.x*param.tc/60, adimensionalSolution.controlAction(1,:),'LineWidth',1.1); hold on
    % stairs(adimensionalSolution.x*param.tc/60, adimensionalSolution.controlAction(2,:),'LineWidth',1.1);
    % stairs(adimensionalSolution.x*param.tc/60, adimensionalSolution.controlAction(3,:),'LineWidth',1.1);

    grid on;
    title("control Action [LVLH]")
    legend("R-BAR","V-BAR","H-BAR",'Location','best')
    xlabel("Time [min]")
    ylabel("control Action [-]")
    
    subplot(3,1,2)
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,1),'LineWidth',1);
    hold on
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,2),'LineWidth',1);
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,3),'LineWidth',1);
    xlabel("Time [min]")
    ylabel("position [km]")
    title("Controlled Relative Dynamics [LVLH]")
    legend("R-BAR","V-BAR","H-BAR",'Location','best')
    grid on
    
    subplot(3,1,3)
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,4)/param.tc,'LineWidth',1);
    hold on
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,5)/param.tc,'LineWidth',1);
    plot(adimensionalSolution.t*param.tc/60,relDynami(:,6)/param.tc,'LineWidth',1);
    xlabel("Time [min]")
    ylabel("velocity [km/s]")
    title("Controlled Relative Velocity [LVLH]")
    legend("R-BAR","V-BAR","H-BAR",'Location','best')
    grid on



    %% zoom in
