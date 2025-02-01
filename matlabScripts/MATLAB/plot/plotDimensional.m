function [] = plotDimensional(solution,param)

    % INITIALISE THE PARAMETERS FOR RESIZING
    tc = 1/(2*pi/2358720);
    xc = 384400;
    solution.y = solution.y';
    earth = ASTRO("EARTH",[-xc*param.massRatio,0,0]);
    moon  = ASTRO("MOON",[xc*(1-param.massRatio),0,0]);

    % RESIZING OF TARGET TRAJECTORY
    solution.y(:,1:3) = solution.y(:,1:3)*xc;
    solution.y(:,4:6) = solution.y(:,4:6)*xc/tc;
    initialState = solution.y(1,:);

    if 0 % choose this to see the trajectory changing color depending to the position
        c = 1:numel(solution.x);      %# colors
        figure()
        h = surface([solution.y(:,1), solution.y(:,1)], [solution.y(:,2), solution.y(:,2)], [solution.y(:,3), solution.y(:,3)], [c(:), c(:)], 'EdgeColor','flat', 'FaceColor','none','LineWidth',0.9);
        barra = colorbar();
        % nota: la colorbar rappresenta il numero di elementi del tempo (t) / la discretizzazione temporale
        barra.Label.String = 'time [elapsed]';
    else
        if isfield(param,"plotType")
            plot3(solution.y(:,1),solution.y(:,2),solution.y(:,3),param.plotType,'LineWidth',1);
        else
            plot3(solution.y(:,1),solution.y(:,2),solution.y(:,3),'r-','LineWidth',1);
        end
    end
    hold on
    plot3(initialState(1),initialState(2),initialState(3),'ko','LineWidth',1);

    % PLOTTING ALSO CHASER IF PRESENT
    if size(solution.y,2) > 6
        solution.y(:,7:9) = solution.y(:,7:9)*xc;
        solution.y(:,10:12) = solution.y(:,10:12)*xc/tc;
        initialState = solution.y(1,:);
        hold on
        plot3(solution.y(:,7),solution.y(:,8),solution.y(:,9),'b-','LineWidth',1);
        plot3(initialState(7),initialState(8),initialState(9),'kv','LineWidth',1);

        %earth.plotAstro;
        moon.plotAstro;
        legend("Target Dynamics","Target Initial Position","Chaser Dynamics","Chaser Initial Position",'location','best')
    else
        earth.plotAstro;
        moon.plotAstro; 
        legend("Target Dynamics","Target Initial Position","","",'location','best')
    end

    
    grid on
    view(30,30)
    xlabel("X [km]");
    ylabel("Y [km]");
    zlabel("Z [km]");
    axis equal
end