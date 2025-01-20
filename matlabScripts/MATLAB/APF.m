function [controlAction,sigma] = APF(relativeState_L,constraintType,param)

    % coefficients definition
    adi2meters = param.xc*1e3;
    Umax = param.maxAdimThrust;

    % extraction of relative state and conversion to meters and meters per second
    rho =   relativeState_L(1:3)*adi2meters;
    v_rho = relativeState_L(4:6)*adi2meters/param.tc;

    switch upper(constraintType)
        case 'SPHERE' 
            % constraints characteristic dimensions definition
            % SphereRadius_KOS = 200; %[m]
            % SphereRadius_AS = 1e3; % [m]
            SphereRadius_SS = 2.5e3; %[m]

            % coefficients definition
            K_SS_inside  = [1e2;5e3;1e2];
            K_SS_outside = [1e5;5e5;1e5];

            % potential field computation
            if (rho'*rho-SphereRadius_SS^2<=0) % if constraint is violated
                NablaUrep_SS = -rho./norm(rho);
                NablaU_APF = K_SS_inside.*NablaUrep_SS; % inside the sphere
            else
                gamma = @(r,constRadius) abs(r(1)^2 + r(2)^2 + r(3)^2 - constRadius^2); %% 
                NablaGamma = 2*rho(:);
                deltaro = (rho-(param.holdingState(1:3)*adi2meters));
                NablaU_APF = K_SS_outside.*(deltaro/gamma(rho,SphereRadius_SS)^2 - deltaro'*deltaro*NablaGamma/gamma(rho,SphereRadius_SS)^3); % 
            end   

        case 'CONE'
            % constraints characteristic dimensions definition
            acone = 0.04; % note: these are adimensional parameters to have 0.4m of radius at docking port
            bcone = 10;   % note: these are adimensional parameters to have 0.4m of radius at docking port
            
            % coefficients definition
            % K_C_inside = [.1; 1e-1; .1]; % era : 1e-1
            K_C_inside = [1; 0; 1] + [1.5; 5e-1; 1.5].*(abs(rho(2))^3/(1e9));
            K_C_outside = [1e1; 0; 1e1];
            
            % approach cone definition
            h = @(r) (r(1)^2 + acone^2*(r(2)-bcone)^3 + r(3)^2);
        
            % computation of the nablas
            Nablah = @(r) [2*r(1); 3*acone^2*(r(2)-bcone)^2; 2*r(3)];
         
            if (rho(1)^2+rho(3)^2>=-(acone^2*(rho(2)-bcone)^3)) % if constraint is violated
                NablaU_APF = K_C_outside.*(rho./norm(rho));
            else
                NablaU_APF = K_C_inside.*( rho./h(rho).^2 - (rho'*rho)*Nablah(rho)./h(rho).^3 );
            end                             

        otherwise
            error("Constraint not defined properly.");
    end

    % sliding surface definition and control action computation
    sigma = NablaU_APF;
    controlAction = -Umax*tanh(sigma);

end