close all
clc
clear

tt = [-pi:0.1:pi];
delta(1,:) = 10*sin(tt);
delta(2,:) = 10*cos(tt);
delta(3,:) = 0;
factor = @(delta) (1+1*(0.5+delta(2)./norm(delta)/2)^(4))

value = [];
for i = 1:length(delta(2,:))
    value(end+1) = factor(delta(:,i));
end

figure
plot(tt,value)

figure
plot3(delta(1,:),delta(2,:),value);
hold on;
plot3(delta(1,1),delta(2,1),delta(3,1),'o');
xlabel("R");
ylabel("V");
zlabel("H")
grid on
axis equal
