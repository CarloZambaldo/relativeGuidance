function [] = plotConstraintsVisualization(DeltaIC_meters,type,colore)
    % coefficients definition
    acone = 0.02;
    bcone = 10;

    if DeltaIC_meters < 200
        rsphere = 200; % [m]
    else
        rsphere = DeltaIC_meters;
    end
    % DeltaIC_meters = min(norm(DeltaIC_meters)*384400*1e3,1e3);
    
    % approach cone definition
    %h = @(r) r(1)^2 + acone^2*(r(2)-bcone)^3 + r(3)^2;
    
    if nargin<2
        type = 'C';
    end

    switch upper(type)

        %% SPHERE %% 
        case 'S'
            z = @(RbarX,VbarX) real(sqrt(rsphere^2-RbarX.^2-VbarX.^2))*1e-3;
            pointsR = linspace(-rsphere,rsphere,501);
            pointsV =  linspace(-rsphere,rsphere,501);
            [X,Y] = meshgrid(pointsR,pointsV);
    
            sferaz = z(X,Y);
            for i = 1:size(X,1)
                for j = 1:size(X,2)
                    if X(i,j)^2+Y(i,j)^2 > (rsphere*1.01)^2
                        sferaz(i,j) = NaN;
                    end
                end
            end
    
            if nargin<3
                colore = 'red';
            end
            
            surf(X*1e-3,Y*1e-3,sferaz,'FaceColor',colore,'FaceAlpha',0.5,'EdgeColor','none');
            hold on
            surf(X*1e-3,Y*1e-3,-sferaz,'FaceColor',colore,'FaceAlpha',0.5,'EdgeColor','none')


        %% CONE %% 
        case 'C'
            z = @(RbarX,VbarX) real(sqrt(acone^2*bcone^3 - 3*acone^2*bcone^2.*VbarX + 3*acone^2*bcone.*VbarX.^2 - acone^2.*VbarX.^3 - RbarX.^2));
            pointsR = linspace(-3*DeltaIC_meters,3*DeltaIC_meters,5501);
            pointsV =  linspace(-DeltaIC_meters,0,1001);
            [X,Y] = meshgrid(pointsR,pointsV);
           
            halfCone = z(X,Y); % [m]
            toll = max(max(diff(pointsR)),max(diff(pointsV)))*1.9;
            for i = 1:size(X,1)
                for j = 1:size(X,2)
                    if -(Y(i,j)-toll-bcone)^3 < ((X(i,j).^2/acone^2))
                        halfCone(i,j) = NaN;
                    end
                end
            end

            % plots is in km, the computation of the cone is in meters
            surf(X*1e-3,Y*1e-3,halfCone*1e-3,'FaceColor','black','FaceAlpha',0.5,'EdgeColor','none','EdgeLighting','gouraud');
            hold on
            surf(X*1e-3,Y*1e-3,-halfCone*1e-3,'FaceColor','black','FaceAlpha',0.5,'EdgeColor','none','EdgeLighting','gouraud');
    end
    xlabel("R-BAR");
    ylabel("V-BAR");
    zlabel("H-BAR");
    axis equal
end