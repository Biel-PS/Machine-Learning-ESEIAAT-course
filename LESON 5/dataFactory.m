classdef dataFactory < handle
    
    properties (Access = private)
        data
        x
    end
    
    methods (Access = public)
        function obj = dataFactory()
            %obj.init(cParams);
        end
        
        function generateDataX(obj,sampleVolume,dataVolume,noiseMag,start)
            obj.data.x = zeros(sampleVolume,dataVolume);
            I = ones(size(obj.data.x,1),1);
            
            if (start == 2)
                obj.data.x(:,1) = 1;
            end
            for i = start:dataVolume
                obj.data.x(:,i) = (rand(size(obj.data.x,1),1)-0.5.*I)*noiseMag;
            end
            obj.data.x(:,3) = obj.data.x(:,3).*10;
            obj.data.xLine = obj.data.x;
            obj.x= obj.data.x;
            obj.data = obj.data;
        end

        function generateDataY(obj,theta,noiseMag,dataSet)
            I = ones(size(obj.x,1),1);
            noiseVec = (rand(size(obj.x,1),1)-0.5.*I)*noiseMag;
            h = zeros(size(obj.x,1),length(theta)/dataSet);
            cont = 1;
            for i= 1:dataSet:(length(theta) -1)
                yLine (:,cont) = obj.x*theta(i:(i+(dataSet-1)))';
                h(:,cont) = yLine(:,cont) + noiseVec;
                
                cont = cont+1;
            end
            data1 =h(:,1) > 0 & h(:,2) > 0 & h(:,3) > 0;  
            data2 = h(:,1) < 0 & h(:,2) < 0 & h(:,3) < 0;
            data3 = h(:,1) < 0 & h(:,2) < 0 & h(:,3) > 0;
          
            obj.data.y = [data1,data2,data3];
            obj.data.yLine = yLine;          
        end

        function splitData (obj,testRatio)
            sampleSize = length(obj.data.x(any(obj.data.y(:,:),2), 2));

            randomSet = randperm(sampleSize);
            sampleSizeTest = round(sampleSize*testRatio);

            randomTest = randomSet(1:sampleSizeTest);
            randomTrain = randomSet((sampleSizeTest+1):end);
                 
            obj.data.train.x(:,:) =  obj.data.x(randomTrain,:);
            obj.data.test.x(:,:) =  obj.data.x(randomTest,:);

            for i = 1:size(obj.data.y,2)
                obj.splitDataFunctional(randomTest,randomTrain,i)
            end
        end

        function reDefineData (obj)
            obj.data.x = obj.data.x(any(obj.data.y(:,:),2),:);
            obj.data.y = obj.data.y(any(obj.data.y(:,:),2),:);
            obj.x = obj.data.x;
        end
        function data = getData (obj)
            data = obj.data;
        end
        
    end
    methods (Access = private)
        function obj = init(obj,cParams)
        end

        function splitDataFunctional (obj,randomTest,randomTrain,index)

           
            obj.data.train.y(:,index) =  obj.data.y(randomTrain,index);
            obj.data.test.y(:,index) =  obj.data.y(randomTest,index);
        end
    end
end

