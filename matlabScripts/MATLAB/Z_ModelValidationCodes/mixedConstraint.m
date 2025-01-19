function F = sphere_with_cone(r, R, acone, bcone)
    % Funzione della sfera
    sphere = r(1)^2 + r(2)^2 + r(3)^2 - R^2;
    
    % Funzione del cono
    cone = r(1)^2 + acone^2 * (r(2) - bcone)^3 + r(3)^2 - R^2;
    
    % Combinazione: scegli sfera o cono
    if r(2) > bcone
        F = cone; % Dentro la regione del cono
    else
        F = sphere; % Fuori dalla regione del cono
    end
end


function visualize_sphere_with_cone(R, acone, bcone, gridSize)
    % Parametri di default
    if nargin < 1, R = 1; end % Raggio della sfera
    if nargin < 2, acone = 1; end % Apertura del cono
    if nargin < 3, bcone = 0; end % Altezza di inizio del cono
    if nargin < 4, gridSize = 50; end % Risoluzione della griglia
    
    % Griglia 3D
    x = linspace(-2*R, 2*R, gridSize);
    y = linspace(-2*R, 2*R, gridSize);
    z = linspace(-2*R, 2*R, gridSize);
    [X, Y, Z] = meshgrid(x, y, z);
    
    % Calcolo della superficie
    F = zeros(size(X));
    for i = 1:numel(X)
        r = [X(i), Y(i), Z(i)];
        F(i) = sphere_with_cone(r, R, acone, bcone);
    end
    
    % Visualizzazione
    figure;
    isosurface(X, Y, Z, F, 0); % Mostra la superficie
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Sfera compenetrata da un cono');
    colormap('autumn');
    lighting gouraud;
    camlight;
end
visualize_sphere_with_cone(1, 1.5, 0.2, 100);
