clear;clc;close all;
noiseMagData = 1;
noiseMagX = 100;
theta = [-0.2,0.8,0.1,0.1,0.1,-0.3,1,0.4,0.1];
dataVolume = 3;

sampleVolume = 100;
trainDataRatio = 0.2;
start = 2;
dataSet = 3;

dataClass = dataFactory();

dataClass.generateDataX(sampleVolume,dataVolume,noiseMagX,start);
dataClass.generateDataY(theta,noiseMagData,dataSet);
dataClass.reDefineData();
testRatio = 0.2;
dataClass.splitData(testRatio);
data = dataClass.getData(); 

x2L1 = -(data.xLine(:,2)*theta(2)+theta(1))/theta(3);
x2L2 = -(data.xLine(:,2)*theta(5)+theta(4))/theta(6);
x2L3 = -(data.xLine(:,2)*theta(8)+theta(7))/theta(9);

figure();

hold on

scatter(data.test.x(data.test.y(:,1), 2), data.test.x(data.test.y(:,1), 3), 50, 'g', 'DisplayName', 'data1 test');
scatter(data.test.x(data.test.y(:,2), 2), data.test.x(data.test.y(:,2), 3), 50, 'r', 'DisplayName', 'data2 test');
scatter(data.test.x(data.test.y(:,3), 2), data.test.x(data.test.y(:,3), 3), 50, 'y', 'DisplayName', 'data3 test');

scatter(data.train.x(data.train.y(:,1), 2), data.train.x(data.train.y(:,1), 3), 50, 'g', 'filled', 'DisplayName', 'data1 trial');
scatter(data.train.x(data.train.y(:,2), 2), data.train.x(data.train.y(:,2), 3), 50, 'r', 'filled', 'DisplayName', 'data2 trial');
scatter(data.train.x(data.train.y(:,3), 2), data.train.x(data.train.y(:,3), 3), 50, 'y', 'filled', 'DisplayName', 'data3 trial');

scatter (data.xLine(:,2),x2L1,'b', 'filled', 'DisplayName', 'data1')
scatter (data.xLine(:,2),x2L2,'g', 'filled', 'DisplayName', 'data2')
scatter (data.xLine(:,2),x2L3,'r', 'filled', 'DisplayName', 'data3')

hold off
title("X1 vs X2");
xlabel("X1")
ylabel("X2")
legend();

s.networkGeometry = 3;
s.numInput = 3;
s.x = data.train.x;
s.data = data;
s.lam = 1;

network = Network(s);

theta = [0.1 0.1 0.1 0.2 0.2 0.2 0.3 0.3 0.3];

sS.ClassGradient = network;
sS.tol = 5e-4;

solver = Solver(sS);
tau = 1e-5; 
thetaResult = solver.gradientSolver(tau,theta);
thetaResult = reshape(thetaResult,[],1);

x2L1 = -(data.xLine(:,2)*thetaResult(2)+thetaResult(1))/thetaResult(3);
x2L2 = -(data.xLine(:,2)*thetaResult(5)+thetaResult(4))/thetaResult(6);
x2L3 = -(data.xLine(:,2)*thetaResult(8)+thetaResult(7))/thetaResult(9);

figure();
hold on
scatter(data.test.x(data.test.y(:,1), 2), data.test.x(data.test.y(:,1), 3), 50, 'g', 'DisplayName', 'data1 test');
scatter(data.test.x(data.test.y(:,2), 2), data.test.x(data.test.y(:,2), 3), 50, 'r', 'DisplayName', 'data2 test');
scatter(data.test.x(data.test.y(:,3), 2), data.test.x(data.test.y(:,3), 3), 50, 'y', 'DisplayName', 'data3 test');

scatter(data.train.x(data.train.y(:,1), 2), data.train.x(data.train.y(:,1), 3), 50, 'g', 'filled', 'DisplayName', 'data1 trial');
scatter(data.train.x(data.train.y(:,2), 2), data.train.x(data.train.y(:,2), 3), 50, 'r', 'filled', 'DisplayName', 'data2 trial');
scatter(data.train.x(data.train.y(:,3), 2), data.train.x(data.train.y(:,3), 3), 50, 'y', 'filled', 'DisplayName', 'data3 trial');

scatter (data.xLine(:,2),x2L1,'b', 'filled', 'DisplayName', 'data1')
scatter (data.xLine(:,2),x2L2,'g', 'filled', 'DisplayName', 'data2')
scatter (data.xLine(:,2),x2L3,'r', 'filled', 'DisplayName', 'data3')
hold off
legend();