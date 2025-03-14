clear;clc;close all;
noiseMagData = 0.05;
noiseMagX = 1;
theta = [-0.2,0.8,0.9,0.5];
dataVolume = 4;

sampleVolume = 500;
trainDataRatio = 0.8;
start = 2;

data.x = zeros(sampleVolume,dataVolume);

I = ones(size(data.x,1),1);

if (start == 2)
    data.x(:,1) = 1;
end

for i = start:dataVolume
    data.x(:,i) = (rand(size(data.x,1),1)-0.5.*I)*noiseMagX;
end
data.x(:,end) = data.x(:,2).^2;

[data.y, data.h] = createData (data.x,noiseMagData,theta);
data = splitData (trainDataRatio,data);


tau = 5e-6; 
tol = 1e-9;

initialGuess = theta + 0.05;
% 
[thetaResult,cost] = solver (data,tau,tol,initialGuess);
% 
x1 = 2;
x2 = 3;

funResult = -(thetaResult(1) + thetaResult(2).*data.x(:,2) + thetaResult(4).*data.x(:,4))./thetaResult(3);
generated = -(theta(1) + theta(2).*data.x(:,2) + theta(4).*data.x(:,4))./theta(3);

figure();
hold on
colors = [data.y, 1-data.y, zeros(size(data.y))];

% 
plot (data.x(:,2),funResult,DisplayName='Boundary comptued')
plot (data.x(:,2),generated,DisplayName='Boundary of generated data')

scatter(data.train.x(data.train.y, x1), data.train.x(data.train.y, x2), 50, 'g', 'filled', 'DisplayName', 'y = 1, train');
scatter(data.train.x(~data.train.y, x1), data.train.x(~data.train.y, x2), 50, 'r', 'filled', 'DisplayName', 'y = 0, train');
scatter(data.test.x(data.test.y, x1), data.test.x(data.test.y, x2), 50, 'y', 'filled', 'DisplayName', 'y = 1, test');
scatter(data.test.x(~data.test.y, x1), data.test.x(~data.test.y, x2), 50, 'b', 'filled', 'DisplayName', 'y = 0, test');
hold off
title("X1 vs X2");
xlabel("X1")
ylabel("X2")
legend();



function [Handside, h] = createData (x,noiseMag,theta)
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

function h = computeH(theta,x)
    h = x*theta' ;
end

function sig = sigmoid(x)
    
    sig = 1 ./(1+exp(-x));

end

function grad = sigGrad(g)
    I = ones(size(g,1),1);
    grad= g.*(I-g);

end

function cost = lReg (y,g)
    n = length(y);
    I = ones(length(g),1);
    op = (I-y).*(-log((I-g)+1e-20)) + y.*(-log(g+1e-20));
    cost = 1/n .* sum(op);
end

function grad = costGradient (x,y,g)
    n = length(y);
    op = (g-y).*x;
    grad = 1/n .* sum(op,1); %el sumatori del gradient als apunts, que vol dir????
end

function [theta,cost] = solver (data,tau,tol,initialGuess)
    
error = 1;

theta = initialGuess; 
xt = data.train.x;
y = data.train.y;
cost = [1];
    while (error > tol)
        % h = computeH(theta,xt);
        % y = sigmoid(h);
        g = getHandside(theta,xt);
        %error = lReg(y,g);
       
        grad = costGradient(xt,y,g);
        error = norm(grad);
        theta = theta - grad.*tau;
        fprintf('Error: %d \n',error)
        
        if (cost(end) ~= error)
            cost(1+end) = error;
        end
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
