clear;clc;close all;
noiseMagData = 0.05;
noiseMagX = 1;
theta = [0.1,0.2];
dataVolume = 2;

sampleVolume = 200;
trainDataRatio = 0.8;


data.x = zeros(sampleVolume,dataVolume);

I = ones(size(data.x,1),1);
for i = 1:dataVolume
    data.x(:,i) = (rand(size(data.x,1),1)-0.5.*I)*noiseMagX;
end

data.y = createData (data.x,noiseMagData,theta);
data = splitData (trainDataRatio,data);

figure();
hold on
colors = [data.y, 1-data.y, zeros(size(data.y))];
%scatter (data.x(:,1),data.x(:,2),50,'filled',colors,DisplayName= "training data");
scatter(data.train.x(data.train.y, 1), data.train.x(data.train.y, 2), 50, 'g', 'filled', 'DisplayName', 'y = 1, train');
scatter(data.train.x(~data.train.y, 1), data.train.x(~data.train.y, 2), 50, 'r', 'filled', 'DisplayName', 'y = 0, train');
scatter(data.test.x(data.test.y, 1), data.test.x(data.test.y, 2), 50, 'y', 'filled', 'DisplayName', 'y = 1, test');
scatter(data.test.x(~data.test.y, 1), data.test.x(~data.test.y, 2), 50, 'b', 'filled', 'DisplayName', 'y = 0, test');
hold off
title("X1 vs X2");
xlabel("X1")
ylabel("X2")
legend();

tau = 1e-1; 
tol = 1e-9;
initialGuess = [0.1,0.1];

[theta,cost] = solver (data,tau,tol,initialGuess);

function Handside = createData (x,noiseMag,theta)
    I = ones(size(x,1),1);
    noiseVec = (rand(size(x,1),1)-0.5.*I)*noiseMag;
    h = x*theta' + noiseVec;
    
    Handside = h(:)>=0;
    % Handside(~Handside) = 1e-8;
    % Handside(Handside) = 1+1e-8;

    
end

function y = getHandside(theta,x)
    h = x*theta' ;
    y = h(:)>=0;

end

function cost = lReg (y,g)
    n = length(y);
    I = ones(length(g),1);
    op = (I-y).*(-log((I-g)+1e-20)) + y.*(-log(g+1e-20));
    cost = 1/n .* sum(op);
end

function grad = costGradient (x,y,g)
    n = length(y);
    op = x'*(g-y);
    grad = 1/n .* sum(op);
end

function [theta,cost] = solver (data,tau,tol,initialGuess)
    
error = 1;

theta = initialGuess; 
xt = data.train.x;
yt = data.train.y;
g = data.train.y;
cost = [];
    while (error > tol)
        y = getHandside(theta,data.train.x);
        error = lReg(y,g);
        grad = costGradient(xt,y,g);
        theta = theta - grad*tau;
        fprintf('Error: %d \n',error)
        cost(1+end) = error;

    end
end

function data = splitData (testRatio,data)
    sampleSize = length(data.y);
    randomSet = randperm(sampleSize);
    sampleSizeTest = round(sampleSize*testRatio);
    
    randomTest = randomSet(1:sampleSizeTest);
    randomTrain = randomSet((sampleSizeTest+1):end);
    
    data.train.x = data.x(randomTest,:);
    data.train.y = data.y(randomTest);
    
    data.test.x = data.x(randomTrain,:);
    data.test.y = data.y(randomTrain);
end
