clear
close all
clc
clear global OBoptimalTrajectory
global trigger OBoptimalTrajectory 


%%
rng;
n_samples = 1;
n_samples_speed = 1;

phaseID = 2;
tspan = [0 0.025];
triggerReset = 0; % <<<<

% initialization
[param,~,~,~] = initializeSimulation(phaseID,tspan);clc

fprintf("Generating a population for the simulations.\n");

%% target positions
initialStateTarget_S_batch = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]';

%% uniform distribution
n_ICs = n_samples * n_samples_speed;

n_targets_pos = size(initialStateTarget_S_batch,2);
n_population = n_ICs + n_targets_pos - 1;

fprintf("Starting Monte Carlo analysis... (this operation can take several minutes)\n");
tic

% population INITIALIZATION
index_pop = 0;
POP = zeros(6,n_population);

%% generate the population (states)
switch 2
    case 1
        val.R_BAR = -0.5 + 1*rand(1,n_samples);  % form -0.5 to +0.5 km
        val.V_BAR = -4 + 2*rand(1,n_samples);  % form -4 to -2 km
        val.H_BAR = -0.5 + 1*rand(1,n_samples);  % form -0.5 to +0.5 km
        val.speed_R_BAR = 1e-3*( -2 + 4*rand(1,n_samples_speed) ); % rand out in m/s, result in km/s
        val.speed_V_BAR = 1e-3*( -2 + 4*rand(1,n_samples_speed) ); % rand out in m/s, result in km/s
        val.speed_H_BAR = 1e-3*( -2 + 4*rand(1,n_samples_speed) ); % rand out in m/s, result in km/s
        for index_R = 1:length(val.R_BAR)
            for index_V = 1:length(val.V_BAR)
                for index_H = 1:length(val.H_BAR)
                    for index_speed_R = 1:length(val.speed_R_BAR)
                        for index_speed_V = 1:length(val.speed_V_BAR)
                            for index_speed_H = 1:length(val.speed_H_BAR)
                                index_pop = index_pop + 1;
                                POP(:,index_pop) = [val.R_BAR(index_R); val.V_BAR(index_V); val.H_BAR(index_H);
                                                    val.speed_R_BAR(index_speed_R); val.speed_V_BAR(index_speed_V); val.speed_H_BAR(index_speed_H)];
                            end
                        end
                    end
                end
            end
        end
    case 2
        val.R_BAR       = -1 + 2*rand(1,n_ICs);  % form -1 to +1 km
        val.V_BAR       = -4 + 2*rand(1,n_ICs);  % form -4 to -2 km
        val.H_BAR       = -1 + 2*rand(1,n_ICs);  % form -1 to +1 km
        val.speed_R_BAR = 1e-3*( -2 + 4*rand(1,n_ICs) ); % rand out in m/s, result in km/s
        val.speed_V_BAR = 1e-3*( -2 + 4*rand(1,n_ICs) ); % rand out in m/s, result in km/s
        val.speed_H_BAR = 1e-3*( -2 + 4*rand(1,n_ICs) ); % rand out in m/s, result in km/s

        POP = [val.R_BAR; val.V_BAR; val.H_BAR; val.speed_R_BAR; val.speed_V_BAR; val.speed_H_BAR];
end

%% RUN THE SIMULATIONS

% adimensionalize the initial conditions
POP = POP./param.xc;
POP(4:6,:) = POP(4:6,:).*param.tc;

% initialize
timeHistory = param.tspan(1):(1/param.freqGNC):param.tspan(2);
fail = zeros(1,n_population);
success = zeros(1,n_population);
trajectory = zeros(length(timeHistory), 12, n_population);
terminalState = zeros(1,6,n_population);


for trgt_id = 1:n_targets_pos
    initialStateTarget_S = initialStateTarget_S_batch(:,trgt_id);

    for sim_id = 1:n_ICs 
        fprintf(" ## RUNNING SIMULATION %d OUT OF %d ##\n",sim_id+trgt_id-1,n_population);
        trigger = triggerReset;
        if trigger == 0
            fprintf("   Simulating SAFE MODE (APF only).\n")
        end
        
        % execute the simulation
        initialRelativeState_L = POP(:,sim_id+trgt_id-1);
        initialRelativeState_S = convert_LVLH_to_S(initialStateTarget_S,initialRelativeState_L,param);
        initialStateChaser_S = initialRelativeState_S + initialStateTarget_S;
    
        fprintf("Initial Distance between C and T: %.2f [km]\n",norm(initialRelativeState_S(1:3))*param.xc);
        fprintf("Initial Relative velocity between C and T: %.2f [m/s]\n\n",norm(initialRelativeState_S(4:6))*param.xc/param.tc*1e3);


        fullInitialState = [initialStateTarget_S; initialStateChaser_S];
        [timeHistory,fullStateHistory,controlActionHistory_L,info] = fullSimulationFunction(fullInitialState,param);
        
        trajectory(:,:,sim_id+trgt_id-1) = fullStateHistory;
        fail(sim_id+trgt_id) = info.crashedBool;
        success(sim_id+trgt_id-1) = info.aimReachedBool;
    end
end

%%
failRate = sum(fail)/(n_population)*100;

fprintf("FAIL RATE: %.2f%%\n",failRate);

cic = toc;
fprintf("Monte Carlo analysis terminated in %g [min].\n",cic/60);


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


% % % % %%
% % % % for sim_id = 1:n_population
% % % %     fprintf(" PLOT %d OUT OF %d\n",sim_id,n_population)
% % % % 
% % % %     soluz = trajectory(:,:,sim_id);
% % % % 
% % % %     indicezeri = soluz(:,7:12) == 0;
% % % %     soluz(indicezeri,1:6) = 0;
% % % % 
% % % %     indiceValori = ~(soluz(:,1) == 0 & soluz(:,2) == 0 & soluz(:,3) == 0);
% % % %     soluz = soluz(indiceValori,:);
% % % %     time = timeHistory(indiceValori);
% % % % 
% % % %     relDynami = zeros(length(time),6);
% % % % 
% % % %     for id = 1:length(time)
% % % %         [rotatedRelativeState] = convert_S_to_LVLH(soluz(id,1:6)',soluz(id,7:12)'-soluz(id,1:6)',param);
% % % %         relDynami(id,1:6) = rotatedRelativeState;
% % % %     end
% % % % 
% % % %     terminalState(:,:,sim_id) = relDynami(end,1:6);
% % % % 
% % % % end

%
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
lim = 1.25*max(Rmax,Hmax);
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
lim = 1.25*max(Rmax,Vmax);
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
lim = 1.25*max(sRmax,sHmax);
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
lim = 1.25*max(sRmax,sVmax);
xlim([-.1, .1])
ylim([-.1, .1])
xlabel("V-BAR [m/s]");
ylabel("R-BAR [m/s]");
title("velocity")