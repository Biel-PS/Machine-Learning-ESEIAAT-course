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
            obj.neuron = Neuron(cParams);

        end
        function g = computeResults(obj,theta)
            nNum = obj.networkGeometry;
            th = reshape(theta,nNum,[]);
            g = (obj.neuron.computeNeuron(th))';            
        end

        function grad = computeGradient(obj,g)
           nNum = obj.networkGeometry;
           grad = zeros(nNum,nNum);
           for i = 1:nNum
               Y = obj.data.train.y(:,i);
               G = g(:,i);
               X = obj.x(:,:);
               grad(:,i) = obj.gradFun(G,Y,X)';
           end
        end

        function grad = solve(obj,theta)
           g = obj.computeResults(theta);
           grad = obj.computeGradient(g);
        end
    end
    methods (Access = private)
        function init (obj,cParams)
            obj.networkGeometry = cParams.networkGeometry;
            obj.numInput = cParams.numInput;
            obj.x =  cParams.x;
            obj.lam = cParams.lam;
            obj.data = cParams.data;
        end
        function J = costFun (obj,g)
            
             m = length(obj.data.train.y(:,1));
             k = length(obj.data.train.y(:,2));
             y = obj.data.y;
             I = ones(m,k);
    
             op = (I-y).*(-log((I-g))) + y.*(-log(g)) + obj.lam.*norm(theta).^2;
             J = 1/m .* sum(sum(op,2),1);
        end
        function grad = gradFun(obj,g,y,x)

            m = length(obj.data.train.y(:,1));
            op = (g-y).*x;
            grad = 1/m .* sum(op,1);
        end
    end
end

