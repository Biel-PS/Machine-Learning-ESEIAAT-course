clc; clear; close all;

dataClass = DataClass();


folderTrainData = 'dataset_faces_SET2/TRAIN_IMAGES';
folderTestData = 'dataset_faces_SET2/TEST IMAGES';
eigenToShow = 9;
r = 1;
dataClass.solveEigenFaces(folderTrainData,eigenToShow,r,'TRAIN');
dataClass.reconTestEigenFace (folderTestData,r,'TEST')
