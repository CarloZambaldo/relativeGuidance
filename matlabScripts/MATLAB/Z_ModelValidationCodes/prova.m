close all
clc
clear
SphereRadius_AS = 1e3; % [m]

safe_rho = [0;8e3];
p=norm(2e3);

pointsR = linspace(-p,p,501);
pointsV = linspace(-p,p,501);
pointsH = linspace(-p,p,501);
[X,Y] = meshgrid(pointsR, pointsV);

% potential fields computation
gamma = @(r,constRadius) (r(1)^2 + r(2)^2  - constRadius^2); %% + r(3)^2
NablaUrep_AS  = @(rho) (rho-safe_rho)/gamma(rho,SphereRadius_AS)^2  - (rho-safe_rho)'*(rho-safe_rho)*2*rho/gamma(rho,SphereRadius_AS)^3;
NablaUrep_AS = @(rho)  1000*rho./norm(rho).*(rho'*rho-SphereRadius_AS.^2<=0); % 
NN = zeros(size(X));

for i = 1:size(X,1)
    for j = 1:size(X,2)
        NN(i,j) = norm(NablaUrep_AS([X(i,j);Y(i,j)]));
    end
end



figure
surf(X,Y,NN','EdgeColor','none');
xlabel("R-BAR");
ylabel("V-BAR");
%axis equal
grid on


%%
close all
clc
clear
            acone = 0.08; % note: these are adimensional parameters to have 0.9m of radius at docking port
            bcone = 5; % note: these are adimensional parameters to have 0.9m of radius at docking port

p=norm(20); 
ispilon = 0; % [m]

pointsR = linspace(-p,p,501);
pointsV = linspace(-p,p,501);
pointsH = linspace(-p,p,501);
[X,Y] = meshgrid(pointsR, pointsV);

% potential fields computation

NablaUrep = @(rho)  [1;1e-3]*(norm(rho))*((rho(1)^2+rho(2)^2)>=-(acone^2*(ispilon-bcone)^3))
NN = zeros(size(X));

for i = 1:size(X,1)
    for j = 1:size(X,2)
        NN(i,j) = norm(NablaUrep([X(i,j);Y(i,j)]));
    end
end


figure
surf(X,Y,NN','EdgeColor','none');
xlabel("R-BAR");
ylabel("H-BAR");

grid on