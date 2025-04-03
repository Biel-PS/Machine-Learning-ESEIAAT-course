classdef Network < handle

    properties (Access = private)
        networkGeometry
        
        numInput
        x
        lam

        neuron
        data
    end
    
    methods (Access = public)
        function obj = Network(cParams)
            obj.init(cParams)

        end
        function y = computeResults(obj,theta)
            nNum = obj.networkGeometry;


            y = zeros(size(obj.x,1),nNum);

            th = reshape(theta,nNum,[]);

            for i = 1:nNum
                y(:,i) = (obj.neuron.computeNeuron(th(:,i)))';
            end
            
        end

        function computeGradient



        end
    end
    methods (Access = private)
        function init (obj,cParams)
            obj.networkGeometry = cParams.networkGeometry;
            obj.numInput = cParams.numInput;
            obj.x = cParams.x;
            obj.lam = cParams.lam;
            obj.data = cParams.data;
        end
        function J = costFun (obj,g)
            
             m = length(obj.data.train.y(:,1));
             k = length(obj.data.train.y(:,2));
    
             I = ones(m,k);
    
             op = (I-y).*(-log((I-g))) + y.*(-log(g)) + obj.lam.*norm(theta).^2;
             J = 1/m .* sum(sum(op,2),1);
        end
        function grad = gradFun(obj,g)

            m = length(obj.data.train.y(:,1));
            op = (g-y).*obj.data.train.x;
            grad = 1/m .* sum(op,1);
        end
    end
end

