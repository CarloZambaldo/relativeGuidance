
clear
close all
clc

p=10;

pointsR = linspace(-1,p,501);
pointsV = linspace(-4,4,501);

[X,Y] = meshgrid(pointsR, pointsV);
xbar = 5;

Uatt = @(x,y) .5*((x-xbar).^2+y.^2);
Urep = @(x,y) 1e0./sqrt((x).^2+(y).^2);
% Urep = @(x,y) -sqrt(x.^2+y.^2);

Utot = @(x,y) Urep(x,y) + Uatt(x,y);
figure
surf(X,Y,Utot(X,Y),'EdgeColor','none')
grid on
hold on