function [] = plotAdimensional(solution,param)
    
    solution.y = solution.y';
    initialState = solution.y(1,:);
    c = 1:numel(solution.x);      %# colors
    h = surface([solution.y(:,1), solution.y(:,1)], [solution.y(:,2), solution.y(:,2)], [solution.y(:,3), solution.y(:,3)], [c(:), c(:)], 'EdgeColor','flat', 'FaceColor','none','LineWidth',0.9);
    barra = colorbar();
    % nota: la colorbar rappresenta il numero di elementi del tempo (t) / la discretizzazione temporale
    barra.Label.String = 'time [elapsed]';
    hold on
    plot3(initialState(1),initialState(2),initialState(3),'ro','LineWidth',1.5)
    % plot3(-param.massRatio,0,0,'b*','LineWidth',2);
    % plot3(1-param.massRatio,0,0,'k*','LineWidth',2);
    param.moon.plotAstro(1/384400);
    %param.earth.plotAstro(1/384400);
    legend("","Initial Position")
    
    grid on
    view(30,30)
    xlabel("X [-]");
    ylabel("Y [-]");
    zlabel("Z [-]");
    axis equal
end