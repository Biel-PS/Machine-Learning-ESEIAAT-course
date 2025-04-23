% eigenfaces_pod.m
% Script para generar eigenfaces usando POD (SVD)
clc; close all; clear
% Ruta al dataset
folder = 'dataset_faces_SET2/TRAIN_IMAGES'; % ← CAMBIA ESTA RUTA si es necesario
files = dir(fullfile(folder, '*.png'));
numImages = length(files);

% Leer la primera imagen para obtener dimensiones
imgSample = imread(fullfile(folder, files(1).name));
imgSample = im2double(imgSample);
if size(imgSample, 3) == 3 %We WANT a gray scale image, so we make this comprovation. 
    imgSample = rgb2gray(imgSample);
end
[m, n] = size(imgSample);
dataMatrix = zeros(m * n, numImages); %we define main matrix, where every image of the data set is a column.

% Construir la matriz de datos (una columna por imagen)
for i = 1:numImages
    img = imread(fullfile(folder, files(i).name));
    img = im2double(img);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    dataMatrix(:, i) = img(:);
end

% Eliminar la media (centrar los datos)
meanFace = mean(dataMatrix, 2);
X = dataMatrix - meanFace;

% SVD = POD
[U, S, V] = svd(X, 'econ');

% Mostrar primeros 9 eigenfaces
figure('Name','Primeros 9 Eigenfaces','NumberTitle','off');
for i = 1:9
    subplot(3,3,i);
    imagesc(reshape(U(:,i), m, n));
    colormap gray;
    axis image off;
    title(['Eigenface ', num2str(i)]);
end

% Proyectar y reconstruir la primera imagen con r eigenfaces
r = 20;
coeffs = U(:,1:r)' * X(:,1);  % proyección
recon = U(:,1:r) * coeffs + meanFace;

% Mostrar original y reconstruida
figure('Name','Reconstrucción','NumberTitle','off');
subplot(1,2,1);
imshow(reshape(dataMatrix(:,1), m, n));
title('Original');

subplot(1,2,2);
imshow(reshape(recon, m, n));
title(['Reconstruida con r = ', num2str(r)]);