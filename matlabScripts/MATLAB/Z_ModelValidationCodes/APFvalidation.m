clear
close all
clc

    acone = 0.08; % note: these are adimensional parameters to have 0.9m of radius at docking port
    bcone = 5; % note: these are adimensional parameters to have 0.9m of radius at docking port

    p=100;

    pointsR = linspace(-p,p,5);
    pointsV = linspace(-p,p,5);
    pointsH = linspace(-p,p,5);
    [X,Y,Z] = meshgrid(pointsR, pointsV, pointsH);

% % approach cone definition
%     h = @(r) r(1).^2 + acone.^2.*(r(2)-bcone).^3 + r(3).^2;
% 
%     % computation of the nablas
%     Nablah = @(r) [2.*r(1); 3.*acone.^2.*(r(2)-bcone).^2; 2.*r(3)];
%     NablaUrep = ( rho./h(rho).^2 - (rho'.*rho).*Nablah(rho)./h(rho).^3 );

    NAB_R = p*(X./((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^2 - (2.*X.*(X.*(X) + Y.*(Y) + Z.*(Z)))./((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^3);
    NAB_V = p*(Y./((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^2 - (12.*(Y - 5).^2.*(X.*(X) + Y.*(Y) + Z.*(Z)))./(625.*((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^3));
    NAB_H = p*(Z./((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^2 - (2.*Z.*(X.*(X) + Y.*(Y) + Z.*(Z)))./((4.*(Y - 5).^3)./625 + X.^2 + Z.^2).^3);

    normalizer = sqrt(NAB_R.^2+NAB_V.^2+NAB_H.^2);
    % plots is in km, the computation of the cone is in meters
    figure
    quiver3(X,Y,Z,NAB_R./normalizer,NAB_V./normalizer,NAB_H./normalizer,'b','LineWidth',1);
    xlabel("R-BAR");
    ylabel("V-BAR");
    zlabel("H-BAR");
    axis equal
    grid on

    pointsR = linspace(-p,p,51);
    pointsV = linspace(-p,p,51);
    [X,Y] = meshgrid(pointsR, pointsV);
    z = @(RbarX,VbarX) real(sqrt(acone^2*bcone^3 - 3*acone^2*bcone^2.*VbarX + 3*acone^2*bcone.*VbarX.^2 - acone^2.*VbarX.^3 - RbarX.^2));
    halfCone = z(X,Y);

    % plots is in km, the computation of the cone is in meters
    hold on
    surf(X,Y,halfCone,'FaceColor','black','FaceAlpha',0.5,'EdgeColor','none','EdgeLighting','gouraud');
    surf(X,Y,-halfCone,'FaceColor','black','FaceAlpha',0.5,'EdgeColor','none','EdgeLighting','gouraud');