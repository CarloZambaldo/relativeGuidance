function [] = matlabPLOT(env)





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