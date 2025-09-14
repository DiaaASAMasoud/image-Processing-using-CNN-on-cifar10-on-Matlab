# Image Processing with CNN on CIFAR-10 Using MATLAB

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/86/Uni-duisburg-essen-logo-2022.jpg" alt="University of Duisburg-Essen" width="200">
</p>

<p align="center">
  <b>Repository authored by</b><br>
  Diaa Ahmed Masoud  
  <br><br>
  <b>University of Duisburg-Essen</b>
  <br><br>
  <b>Advisors:</b><br>
  Jonathan Liebeton, M.Sc.<br>
  Univ.-Prof. Dr.-Ing. Dirk Söffker
</p>

## Table of Contents

- [Overview](#overview)  
- [Results Showcase](#results-showcase)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  

##  Overview

This project explores image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNN) in MATLAB. It includes several neural network architectures implemented via MATLAB Live Scripts (`.mlx`), varying in depth and complexity. Features:

- Live scripts for different network configurations: 2-CNN Blocks, 3-CNN Blocks, 4-CNN Blocks, 4-CNN Blocks+ Dense Block, and 8-CNN Blocks+ Dense Block.
- A main execution script (`Main.mlx`) to orchestrate training and evaluation over multiple general configurations.
- Script for loading and preprocessing the CIFAR-10 data (`loadtrainingdata.mlx`).
- Script for loading the training configuration `createTrainingOptions.mlx`,
- Experimentation on particular configurations via `testconfigs.mlx`
- Results aggregation and visualisation scripts (`aggregate_results.mlx`, `Scattergraph.mlx`).
-The Best model resulted in an  Accuracy	of 91.93%, a Precision	of 91.962%, a Recall of 91.93% and an F1 Score	of 91.89%
- different testing models are presented in: `Test Output` folder
- An Excel summary of experiments (`experiment_summary.xlsx`) resulted from the `aggregate_results.mlx` file and can be reproduced, given that more tests were done.

## Results Showcase
### Accuarcy graph based on (optimizer,number of Convolution blocks, Batch size, learning Rate) of the expirments 
![Summary](summary.png)

### Layer architecture for the `5convlayers.mlx`
![35 Layers](35%20layers.png)

### confusion Matrix on Training, Validation and Testing data for the best resulting model
![b1](Best%20Result/CM%20for%20thre%20best%20result.png)
### training curves
![b2](Best%20Result/AccuarcyCurves.png).
### Loss Curves
![b3](Best%20Result/LossCurves.png).

## Getting Started

### Prerequisites
- MATLAB R2018a or later
- Toolboxes: Deep Learning, Image Processing  
- (Optional) Parallel Computing Toolbox for GPU acceleration

### Installation
```bash
git clone https://github.com/deee2o69/image-Processing-using-CNN-on-cifar10-on-Matlab.git
cd image-Processing-using-CNN-on-cifar10-on-Matlab
```
## Usage
- From the `Main.mlx` file, you can directly set the range of settings and optimisation parameters to be looped through
```matlab
Method          = ["sgdm","adam","rmsprop"];
initLearnRate   = [0.1 0.01 0.001];
MaxEpochs       = 150;  
MiniBatchSize   = [256 128];
Momentum        = 0.9;
conv            = 4;
```
And for some particular configurations, you can access `testconfigs.mlx` and use your particular configurations, as an example:
```matlab
configs = [
    struct('mm','sgdm','lr',0.2,'mb',256),...
    struct('mm','adam','lr',0.002,'mb',256),...
    struct('mm','sgdm','lr',0.1,'mb',256),...
    struct('mm','rmsprop','lr',0.002,'mb',256),...
    ];
```
- To use the pretrained model, load the file into MATLAB using 
```matlab
load('5CONV_CIFAR10_sgdm_LR2e-01_BS256_M9e-01_E150_ACC89.86.mat', 'net');
```
Sorry for the long name, but each result had to be differentiated one way or another 
- To export the (`experiment_summary.xlsx`), you just have to run the `aggregate_results.mlx` file
- To get the 3d scatter graph, just run the `Scattergraph.mlx` file.
