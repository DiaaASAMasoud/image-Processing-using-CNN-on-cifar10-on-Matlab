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

This project explores image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNN) in MATLAB. It includes several neural network architectures implemented via MATLAB Live Scripts (`.mlx`), varying in depth and complexity, along with a modern training and evaluation pipeline. Features:

- Live scripts for different network configurations: 2-CNN Blocks, 3-CNN Blocks, 4-CNN Blocks, 4-CNN Blocks+ Dense Block, and 8-CNN Blocks+ Dense Block.
- A compact SE-residual network (`conv6layers`) adding Squeeze-and-Excitation and skip connections to the conv stack, the current best model.
- A dual training engine selected by a single `engine` flag: `trainNetwork` (classic, piecewise schedule) or `trainnet` (mixup, label smoothing, cutout, warm-up + cosine schedule, with a custom `softAccuracy` metric).
- Test-time augmentation (`ttaPredict`) that averages predictions over augmented views for an inference-time boost.
- Bayesian hyperparameter optimisation via `BO_main.mlx` (from scratch) and `BO_pretrained.mlx` (fine-tuning the best saved model).
- GPU-aware batch sizing (`findMaxBatch`) that picks the mini-batch from the available GPU memory.
- Checkpointing for long runs, with automatic resume from the latest checkpoint.
- Script for loading and preprocessing the CIFAR-10 data (`loadtrainingdata.mlx`).
- Script for loading the training configuration `createTrainingOptions.mlx`,
- Experimentation on particular configurations via `testconfigs.mlx`, with a looping driver over general configurations in `GridSearch.mlx`.
- Results aggregation and visualisation scripts (`aggregate_results.mlx`, `scatterer.mlx`).
- The Best model (`conv6layers`) resulted in an Accuracy of 93.94%, rising to 95.1% with TTA.
- different testing models are presented in: `Test Output` folder
- An Excel summary of experiments (`experiment_summary.xlsx`) resulted from the `aggregate_results.mlx` file and can be reproduced, given that more tests were done.

## Results Showcase

The best model — `conv6layers` (SE-residual, ~1.5M params) which resulted in an Accuracy of 93.94%, a Precision of 93.95%, a Recall of 93.94% and an F1 Score of 93.93%, rising to 95.1% with TTA.

### confusion Matrix on Training, Validation and Testing data for the best resulting model
![CM](Best%20Result/ConfusionMatrices_Run1.png)

### Test-time augmentation (TTA) confusion matrix
![TTA](Best%20Result/TTAresults.png)

### Layer architecture for `conv6layers`
![Architecture](Best%20Result/LayerConfig.png)

## Getting Started

### Prerequisites
- MATLAB R2026a or later — the `trainnet` engine, cosine schedule objects, and the figure `theme` switch need a recent release; the classic conv-block scripts run on older versions.
- Toolboxes: Deep Learning, Image Processing
- Statistics and Machine Learning Toolbox (for the Bayesian optimisation scripts `BO_main.mlx` / `BO_pretrained.mlx`)
- (Optional) Parallel Computing Toolbox for GPU acceleration

### Installation
```bash
git clone https://github.com/deee2o69/image-Processing-using-CNN-on-cifar10-on-Matlab.git
cd image-Processing-using-CNN-on-cifar10-on-Matlab
```
- Download the CIFAR-10 MATLAB binary version and put the `cifar-10-batches-mat` folder in the project root; this is what `loadtrainingdata.mlx` reads.
- Keep the project root as your MATLAB working folder; the scripts add the architectures with `addpath("layer configs")`.
## Usage

- For a single or repeated configuration, open `testconfigs.mlx` and set the knobs at the top. The `engine` flag picks the training path — `trainNetwork` (classic) or `trainnet` (mixup + cosine):
```matlab
engine    = "trainnet";    % "trainNetwork" (classic) or "trainnet" (mixup + cosine)
net       = "conv6layers"; % architecture function in `layer configs`
mm        = "sgdm";        % optimiser: sgdm | adam | rmsprop
lr        = 0.1;           % initial learning rate (cosine peak under trainnet)
mb        = 128;           % mini-batch size
ep        = 300;           % number of epochs
mo        = 0.9;           % momentum (sgdm only)
l2        = 5e-4;          % L2 weight decay 
ckpt      = true;          % save checkpoints (auto-resume if interrupted)
ckptEvery = 10;            % checkpoint every N epochs
dr        = 80;            % LR drop period in epochs   (trainNetwork only)
lrdf      = 0.1;           % LR drop factor             (trainNetwork only)
vp        = 40;            % validation patience        (trainNetwork only)
```
- To tune hyperparameters, run `BO_main.mlx` (Bayesian search from scratch) or `BO_pretrained.mlx` (load the best saved model and fine-tune it). Set the run at the top:

```matlab
engine         = "trainNetwork";   
Method         = "sgdm";
MaxEpochs      = 40;
BONumofItr     = 15;   % Bayesian optimisation evaluations
RNDSeedsTest   = 2;    % seeds to retrain the best config on
lambda         = 0.2;  % overfitting-penalty weight in the BO objective
MomentumFactor = 0.9;
```
- To use the pretrained model, load the file into MATLAB using 
```matlab
load('Best Result/56CONV_CIFAR10_sgdm_LR1e-01_BS128_M9e-01_E300_ACC93.94.mat', 'net');
```
Sorry for the long name, but each result had to be differentiated one way or another 
- To run test-time augmentation on a loaded `net`, just call `ttaPredict(net, 8)`.
- To export the (`experiment_summary.xlsx`), you just have to run the `aggregate_results.mlx` file
- To get the 3d scatter graph, just run the `scatterer.mlx` file.
