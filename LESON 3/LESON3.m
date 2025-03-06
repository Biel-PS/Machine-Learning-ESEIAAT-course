clear; clc; close all

theta2 = 0.5;
theta1 =0.7;
theta0 = 0.2;
theta = [theta0,theta1,theta2];

noiseMag = 100;
capacity = 2;
sampleSize = 70;

x = zeros(sampleSize,3);
x(:,2) = (-sampleSize/2):1:(sampleSize/2 -1 );
x(:,3) = x(:,2).^2;
x(:,1) = 1;
X = x(:,1:2);

y = setData (sampleSize,theta,x,noiseMag);
data = splitData (0.8,x,y);

minSampleValue = 10;
maxSampleValue = 1000;
%%MSE comparison
errorSampleSize = MSEVarSample (data,minSampleValue,maxSampleValue,theta,noiseMag,capacity);
errorCapacityVariation = MSEVarCapacity (data,1,10)

figure();
hold on
plot (errorSampleSize(:,1),errorSampleSize(:,2),DisplayName='MSE training')
plot (errorSampleSize(:,1),errorSampleSize(:,3),DisplayName='MSE test');
title("Fixed capacity, variational sample size");
xlabel("Number of data points")
ylabel("MSE")
hold off
legend(); grid;

figure();
hold on
plot (errorCapacityVariation(:,1),errorCapacityVariation(:,2),DisplayName='MSE training')
plot (errorCapacityVariation(:,1),errorCapacityVariation(:,3),DisplayName='MSE test');
title("Fixed sample size, variational capacity");
xlabel("Capacity (degree of the polynomial)")
ylabel("MSE")
hold off
legend(); grid;

%%Linear and plynomical model comparison
xLinear = [ones(size(data.train.x)),data.train.x];
yLinear = data.train.y;

thLinear = normL2(xLinear,yLinear);
funLinear = Functionf(thLinear,X(:,1:2));

xHighCapTrain= setCapacity (capacity,data.train.x);
yHighCapTrain = data.train.y;
xHightCap = setCapacity (capacity,X(:,2));

thLargeCapacity = normL2(xHighCapTrain,yHighCapTrain);
funLargeCapacity = Functionf(thLargeCapacity,xHightCap);

figure ();
hold on
scatter (data.train.x,data.train.y,"blue",DisplayName= "training data");
scatter (data.test.x,data.test.y,"red",DisplayName= "test data");
plot (X(:,2),funLinear,DisplayName='Lineal')
plot (X(:,2),funLargeCapacity,DisplayName='Overfitting');

hold off
legend();grid

function error = MSEVarSample (data,initVal,maxVal,thetaInitial,noiseMag,capacity)
    
    error = zeros ((maxVal-initVal),3); %training and test MSE
    
    for i = initVal:1:maxVal
        x = zeros(i,3);

        lim =  (i/2);
        x(:,2) = (-lim):1:(lim-1);
        x(:,3) = x(:,2).^2;
        x(:,1) = 1;
       
        y = setData (i,thetaInitial,x,noiseMag);
        
        data = splitData (0.8,x,y);
    
        xTrain= setCapacity (capacity,data.train.x);
        yTrain = data.train.y;

        xTest = setCapacity (capacity,data.test.x);
        yTest = data.test.y;
      
        th = normL2(xTrain,yTrain);

        funEvalTrainingCase= Functionf(th,xTrain);
        funEvalTestingCase= Functionf(th,xTest);
            
        MSEtrain = MSE(yTrain,funEvalTrainingCase);
        MSEtest = MSE(yTest,funEvalTestingCase);

        error(i,:) = [i,MSEtrain,MSEtest];
    end  

end



function error = MSEVarCapacity (data,minCapacity,maxCapacity)

    error = zeros ((maxCapacity-minCapacity),3); %training and test MSE
    
    for i = minCapacity:1:maxCapacity
    
    
        xTrain= setCapacity (i,data.train.x);
        yTrain = data.train.y;

        xTest = setCapacity (i,data.test.x);
        yTest = data.test.y;
      
    
        thLargeCapacity = normL2(xTrain,yTrain);

        funEvalTrainingCase= Functionf(thLargeCapacity,xTrain);
        funEvalTestingCase= Functionf(thLargeCapacity,xTest);
            
        MSEtrain = MSE(yTrain,funEvalTrainingCase);
        MSEtest = MSE(yTest,funEvalTestingCase);

        error(i,:) = [i,MSEtrain,MSEtest];
    end  
end

function y = setData (sampleSize,theta,x,noiseMag)
    noiseVec = zeros(sampleSize,1);
    for i = 1:1:sampleSize
        noiseVec(i) = noiseMag*rand;
    end
    y = x*theta' + noiseVec;
end

function theta = normL2 (x,y)
   theta = ((x.')*x)\(x.')*y;
end

function theta = normL1 (x,y)
    t = ones (size(x,1),1);
    fun = [0;0;t];
    identity = eye(size(x,1));
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2)
end
function theta = normLinf (x,y)
    fun = [0;0;1];
    identity = ones(size(x,1),1);
    A = [x,-identity; -x,-identity];
    b = [y;-y];
    th = linprog(fun,A,b);
    theta = th(1:2);
end


function f = Functionf (theta,x)
   f = x*theta;
end

function data = splitData (testRatio,x,y)
    sampleSize = size(y,1);
    
    randomSet = randperm(sampleSize);
    
    sampleSizeTest = round(sampleSize*testRatio);
    
    randomTest = randomSet(1:sampleSizeTest);
    randomTrain = randomSet((sampleSizeTest+1):end);
    
    data.train.x = x(randomTest,2);
    data.train.y = y(randomTest);
    
    data.test.x = x(randomTrain,2);
    data.test.y = y(randomTrain);
end

function X = setCapacity (n,x) %NO CORSSED TERMS
    prodMatrix = 0:1:(n+1);
    
    X = zeros (size(x,1),size(prodMatrix,2));
    for i = 1:size(prodMatrix,2)
        X(:,i) = x(:,1);
    end
    X = X.^prodMatrix;
end


function error = MSE (dataY, fun)
    error = norm(dataY-fun)/size(dataY,1);
end