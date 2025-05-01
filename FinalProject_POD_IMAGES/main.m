clc; clear; close all;

dataClass = DataClass();
imagePath = 'Grid_6x6.png';
outputDir = 'dataset_faces_SET3';
gridSize = 6;

dataClass.imagePreProcess(imagePath,outputDir,gridSize) 


folderTrainData = 'dataset_faces_SET2/TRAIN_IMAGES';
folderTestData = 'dataset_faces_SET2/TEST IMAGES';
eigenToShow = 9;
r = 20;
dataClass.solveEigenFaces(folderTrainData,eigenToShow,r,'TRAIN');
dataClass.reconTestEigenFace (folderTestData,r,'TEST')
