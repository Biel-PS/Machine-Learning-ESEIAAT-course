clear;clc;close all;
noiseMagData = 0;
noiseMagX = 10;
theta = [-0.2,0.8,0.1,0.1,0.1,-0.3,1,0.4,0.1];
dataVolume = 3;

sampleVolume = 1000;
trainDataRatio = 0.8;
start = 2;
dataSet = 3;

dataClass = dataFactory();

data = dataClass.generateDataX(sampleVolume,dataVolume,noiseMagX,start);
data = dataClass.generateDataY(theta,noiseMagData,dataSet);


x2L1 = (data.x(:,2)*theta(2)+theta(1))/theta(3);
x2L2 = (data.x(:,2)*theta(5)+theta(4))/theta(6);
x2L3 = (data.x(:,2)*theta(8)+theta(7))/theta(9);

figure();
hold on

scatter(data.x(data.y(:,1), 2), data.x(data.y(:,1), 3), 50, 'g', 'filled', 'DisplayName', 'top');
scatter(data.x(data.y(:,2), 2), data.x(data.y(:,2), 3), 50, 'r', 'filled', 'DisplayName', 'bottom');
scatter(data.x(data.y(:,3), 2), data.x(data.y(:,3), 3), 50, 'y', 'filled', 'DisplayName', 'mid');

scatter (data.xLine(:,2),x2L1,'b')
scatter (data.xLine(:,2),x2L2,'green')
scatter (data.xLine(:,2),x2L3,'yellow')

hold off
title("X1 vs X2");
xlabel("X1")
ylabel("X2")
legend();
