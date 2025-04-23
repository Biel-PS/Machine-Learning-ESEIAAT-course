classdef DataClass < handle

    
    properties (Access = private)
    Umat
    Smat
    Vmat
    end
    
    methods (Access = public)
        function obj = DataClass(cParams)
            
        end
        
        function outputArg = method1(obj,inputArg)

        end

        function imagePreProcess (obj,imagePath,outputDir,gridSize)

            img = imread(imagePath);
            if size(img,3) == 3
                img = rgb2gray(img);  % Convert to a gray scale
            end
            img = im2double(img);  % Convert to double

            [fullHeight, fullWidth] = size(img); % Get whole image pixel dimensions
            cellHeight = floor(fullHeight / gridSize); % find individual images
            cellWidth = floor(fullWidth / gridSize);
            
            if ~exist(outputDir, 'dir') %Create a folder to store images
                mkdir(outputDir);
            end
            
            % Cut every image from the whole, to the individual files
            counter = 1;
            for row = 0:grid_size-1
                for col = 0:grid_size-1
                    y1 = row * cellHeight + 1;
                    y2 = y1 + cellHeight - 1;
                    x1 = col * cellWidth + 1;
                    x2 = x1 + cellWidth - 1;
                    
                    subimg = img(y1:y2, x1:x2);
                    filename = fullfile(outputDir, sprintf('face_%d.png', counter));
                    imwrite(subimg, filename);
                    fprintf('Saved: %s\n', filename);
                    counter = counter + 1;
                end
            end
        end

        function solveEigenFaces(obj,folderTrainData,eigenToShow,r,label)

            [dataMatrix,X,m,n,meanFace] = obj.obtainDataFromFigures(folderTrainData);
            % SVD = POD
            [U, S, V] = svd(X, 'econ');
            obj.Umat = U;
            obj.Smat = S;
            obj.Vmat = V;
            % Show eigenFaces
            figure('Name','Eigenfaces','NumberTitle','off');
            for i = 1:eigenToShow
                subplot(3,3,i);
                imagesc(reshape(U(:,i), m, n));
                colormap gray;
                axis image off;
                title(['Eigenface ', num2str(i)]);
            end
            
            % Project and reconstruct with r eigenfaces
            recon = obj.projectInSpace (U,X,meanFace,r);
            % Mostrar original y reconstruida
            obj.plotFaces(dataMatrix,recon,m,n,r,label);
        end

        function reconTestEigenFace (obj,folderTestData,r,label)
            [dataMatrix,X,m,n,meanFace] = obj.obtainDataFromFigures(folderTestData);
            U = obj.Umat;
            recon = obj.projectInSpace (U,X,meanFace,r);
            obj.plotFaces (dataMatrix,recon,m,n,r,label);
        end
    end
    methods (Access = private)
        function init (obj,cParams)

        end

        function recon = projectInSpace (obj,U,X,meanFace,r)
            coeffs = U(:,1:r)' * X(:,1);  % proyección
            recon = U(:,1:r) * coeffs + meanFace;
        end

        function [dataMatrix,X,m,n,meanFace] = obtainDataFromFigures(obj,folder)
           
            files = dir(fullfile(folder, '*.png'));
            numImages = length(files);
            
            % Obtain dimensions of images by reading the first one. 
            imgSample = imread(fullfile(folder, files(1).name));
            imgSample = im2double(imgSample);

            if size(imgSample, 3) == 3
                imgSample = rgb2gray(imgSample);
            end

            [m, n] = size(imgSample);
            dataMatrix = zeros(m * n, numImages);
            
            % Build data matrix (column matrix)
            for i = 1:numImages
                img = imread(fullfile(folder, files(i).name));
                img = im2double(img);
                if size(img, 3) == 3
                    img = rgb2gray(img);
                end
                dataMatrix(:, i) = img(:);
            end
            
            % Extract the mean face from the whole dataset
            meanFace = mean(dataMatrix, 2);
            X = dataMatrix - meanFace;
        end
        function plotFaces (obj,dataMatrix,recon,m,n,r,label)
            figure('Name',label,'NumberTitle','off');
            subplot(1,2,1);
            imshow(reshape(dataMatrix(:,1), m, n));
            title('Original');
            
            subplot(1,2,2);
            imshow(reshape(recon, m, n));
            title(['Reconstructed with r = ', num2str(r)]);
        end
    end
end

