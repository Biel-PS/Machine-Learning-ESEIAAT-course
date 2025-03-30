classdef dataFactory < handle
    
    properties (Access = private)
        data
        x
    end
    
    methods (Access = public)
        function obj = dataFactory()
            %obj.init(cParams);
        end
        
        function data = generateDataX(obj,sampleVolume,dataVolume,noiseMag,start)
            data.x = zeros(sampleVolume,dataVolume);
            I = ones(size(data.x,1),1);
            
            if (start == 2)
                data.x(:,1) = 1;
            end
            for i = start:dataVolume
                data.x(:,i) = (rand(size(data.x,1),1)-0.5.*I)*noiseMag;
            end
            data.x(:,3) = data.x(:,3).*10;
            %data.x(:,end) = data.x(:,2);
            data.xLine = data.x;
            obj.x= data.x;
            obj.data = data;
        end

        function data = generateDataY(obj,theta,noiseMag,dataSet)
            data = obj.data;
            I = ones(size(obj.x,1),1);
            noiseVec = (rand(size(obj.x,1),1)-0.5.*I)*noiseMag;
            h = zeros(size(obj.x,1),length(theta)/dataSet);
            cont = 1;
            for i= 1:dataSet:(length(theta) -1)
                yLine (:,cont) = obj.x*theta(i:(i+(dataSet-1)))';
                h(:,cont) = yLine(:,cont) + noiseVec;
                
                cont = cont+1;
            end
            %data1_new = h(:,1) > 0;
            data1 =h(:,1) > 0 & h(:,2) > 0 & h(:,3) > 0;  
            data2 = h(:,1) < 0 & h(:,2) < 0 & h(:,3) < 0;
            data3 = h(:,1) < 0 & h(:,2) < 0 & h(:,3) > 0;
            % data1 = h(:,:) > 0;  
            % data2 = h(:,:) < 0; :,:
            % data3 = h(:,1) < 0 & h(:,2:length(theta)/2) < 0;
            
            
            data.y = [data1,data2,data3];
            data.yLine = yLine;
            %data.h = [h(data1),h(data2),h(data3)];
            obj.data = data;
             
        end
        function data = splitData (obj,testRatio)
            data = obj.data;
            sampleSize = length(obj.data.y);
            randomSet = randperm(sampleSize);
            sampleSizeTest = round(sampleSize*testRatio);

            randomTest = randomSet(1:sampleSizeTest);
            randomTrain = randomSet((sampleSizeTest+1):end);

            data.train.x = data.x(randomTest,:);
            data.train.y = data.y(randomTest);

            data.test.x = data.x(randomTrain,:);
            data.test.y = data.y(randomTrain);
        end
        
    end
    methods (Access = private)
        function obj = init(obj,cParams)
        end
    end
end

