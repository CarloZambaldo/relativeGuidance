function [param,initialStateTarget_S,initialStateChaser_S,DeltaIC_S] = initializeSimulation(phaseID,tspan,seed)
    %% ADIMENSIONALISERS
    param.xc = 384400;
    param.tc = 1/(2.661699e-6); % (2*pi/2358720);
    
    %% SPACECRAFT PARAMETERS
    
    % CHASER
    param.chaser.mass = 15000;
    param.chaser.Area = 18;
    param.chaser.reflCoeffSpecular = .5;
    param.chaser.reflCoeffDiffuse = .1;

    param.maxAdimThrust = (490/param.chaser.mass)*1e-3/param.xc*param.tc^2; % [adimensional]
    param.holdingState = [0;-4; 0; 0; 0; 0]./param.xc; % [-] [.5;-5; 0; 0; 0; 0]./param.xc; % [-]
    param.dockingState = [0; 0; 0; 0; 0.02e-3*param.tc/param.xc; 0]; % Final relative state similar Luca Thesis
    param.freqGNC = 10*param.tc; % [adimensional Hz]

    % TARGET
    param.target.mass = 40000; % [kg]
    param.target.Area = 110; % [m^2]
    param.target.reflCoeffSpecular = .5;
    param.target.reflCoeffDiffuse = .1;

    %% ENVIRONMENT PARAMETERS
    param.SolarFlux = 1361/299792458; % [W/m^2 / (m/s)]

    %% Simulation Parameters
    param.tspan = tspan;
    param.phaseID = phaseID;
    % This is the delta_state between C and T (applied to T only) at the beginning of the simulation
    % initialStateTarget_S = [1.02134, 0, -0.18162, 0, -0.10176, 9.76561e-07]'; % see PhD thesis
    
    % Astronomical Parameters
    earth = ASTRO("EARTH");
    moon  = ASTRO("MOON");
    
    massEarth = earth.mass;
    massMoon = moon.mass;
    param.massRatio = massMoon/(massEarth+massMoon);
    param.Omega = 2*pi/2358720;
    
    earth = earth.updatePosition([param.massRatio, 0, 0]);
    moon = moon.updatePosition([1-param.massRatio, 0, 0]);
    
    param.moon = moon;
    param.earth = earth;


    %% RANDOMIZED
    if nargin < 3
        rng("shuffle");
        param.rng_settings = rng;
    else
        if strcmp(seed,"any")
             rng("shuffle");
             param.rng_settings = rng;
        elseif isnumeric(seed)
            rng(seed);
            param.rng_settings = rng(seed);
        end
    end
    
    %% FIND THE LUNAR GATEWAY NRHO initialState
    % using the default trajectory
    load("refTraj.mat");
    
    % get random position of the target
    index = randi([2, length(refTraj.y)]);
    initialStateTarget_S = refTraj.y(:,index);

    switch(phaseID)
        case 1 % REACHING SAFE HOLD
            % compute random relative position
            ok = 100;
            while ok > 0 % re-compute random position if it violates the constraint
                rand_position_L = [(-8+16*rand()),(-8+16*rand()),(-8+16*rand())]' / param.xc;
                if norm(rand_position_L)^2 > (1/param.xc)^2
                    ok = -1;
                else
                    ok = ok-1;
                end
            end
            if ok ~= -1
                error("Could not compute the random condition. Please try again with another seed");
            end


            
            rand_velocity_L = (-3+6*rand(3,1)) * 1e-3 / param.xc * param.tc;
            % % %% % rand_position_L = [0;3;0] / param.xc
            % % % % rand_velocity_L = [0;-2;0] * 1e-3 / param.xc * param.tc;
            %%
            % % % % if 1
            % % % %     % using the default trajectory
            % % % %     load("refTraj.mat");
            % % % %     index = randi([2, length(refTraj.y)]);
            % % % %     initialStateTarget_S = refTraj.y(:,index);
            % % % %     initialStateChaser_S = refTraj.y(:,index+26);
            % % % % 
            % % % %     DeltaIC_S = initialStateChaser_S-initialStateTarget_S;
            % % % % else
            % % % %     % also velocity changes:
            % % % %     randoposi = -rndvctr*4/param.xc;%./norm(rndvctr)
            % % % %     randovelo = randmod*1e-3*param.tc/param.xc; % ./norm(randmo)
            % % % %     DeltaIC_S = [randoposi(:); randovelo(:)];
            % % % % end

        case 2 % DOCKING
            % compute random relative position
            rand_position_L = [(-2+4*rand()),(-4+3*rand()),(-2+4*rand())]' / param.xc; % in a radius around the holding point
            rand_velocity_L = (-1+1*rand(3,1)) * 1e-3 / param.xc * param.tc;
        otherwise
            error("No phase id defined");
    end

    % tansform it to Synodic 
    DeltaIC_S = convert_LVLH_to_S(initialStateTarget_S,[rand_position_L; rand_velocity_L],param);

    % compute chaser state
    initialStateChaser_S = initialStateTarget_S + DeltaIC_S;
    
    % compute the initial Relative State in SYNODIC FRAME
    initialRelativeState_S = initialStateChaser_S-initialStateTarget_S;
    if norm(initialRelativeState_S-DeltaIC_S)>=1e-10
        fprintf("value for    DeltaIC: [%f, %f, %f, %f, %f, %f] (norm: %f)\n",DeltaIC_S,norm(DeltaIC_S));
        fprintf("initialRelativeState: [%f, %f, %f, %f, %f, %f] (norm: %f)\n",initialRelativeState_S,norm(initialRelativeState_S));
        fprintf("Error norm: %g\n",norm(initialRelativeState_S-DeltaIC_S))
        error("SOMETHING WENT WRONG IN THE COMPUTATION OF THE RELATIVE STATE!");
    end
    
    fprintf("Initial Distance between C and T: %.2f [km]\n",norm(DeltaIC_S(1:3))*param.xc);
    fprintf("Initial Relative velocity between C and T: %.2f [m/s]\n",norm(DeltaIC_S(4:6))*param.xc/param.tc*1e3);

    
end