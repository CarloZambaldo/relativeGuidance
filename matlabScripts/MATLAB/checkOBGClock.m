clear
close all
clc


%% 
% Initial conditions
state0 = [15; 0]; % x, vx
tspan = [0,100];


E = odeEvent(Direction="both", Response = "callback", EventFcn=@OBClockEvent, CallbackFcn=@changePosition);
F = ode(ODEFcn=@odeFunction,InitialValue=state0,EventDefinition=E,Solver="ode113",RelativeTolerance=1e-10,AbsoluteTolerance=1e-10);
S = solve(F,tspan(1),tspan(2),Refine=1)
t = S.Time;
state = S.Solution;
te = S.EventTime;
ie = S.EventIndex;


% % Define options with event function
% options = odeset('RelTol',1e-13,'Events', @eventFunction);
% 
% % Solve the ODE
% [t, state, te, ye, ie] = ode113(@odeFunction, tspan, state0, options);

figure()
subplot(2,1,1)
plot(te,ie,'r+','LineWidth',1);
title("Event Triggers")
xlabel("time [s]");
ylabel("triggers")
grid on;

subplot(2,1,2)
plot(t,state(1,:),'LineWidth',1);
title("State Dynamics")
xlabel("time [s]");
ylabel("state [m]")
grid on;

figure
semilogy(diff(S.Time))
title("semilog of diff(timeStep)")
xlabel("step id [-]")
ylabel("time step size")
grid on;


%%

function dydt = odeFunction(t, state)
    dydt = [state(2); -9.81];  % Simple falling object with gravity
end

function value = OBClockEvent(t,state)
    f = 10; % [Hz]
    value = sin(pi*f*t); % Monitor when clock goes to 0
end

function [stop,state] = changePosition(t,state)
    stop = false;
    state = state;
end

function [value, isterminal, direction] = eventFunction(t, state)
    f = 1; % [Hz]
    value = sin(pi*f*t); % Monitor when clock goes to 0
    isterminal = 0;        % Stop integration when event occurs
    direction = 0;         % Detect decreasing zeros (falling object)
end

