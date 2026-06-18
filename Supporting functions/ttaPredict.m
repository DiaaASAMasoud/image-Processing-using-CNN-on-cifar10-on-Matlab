function [acc, precision, recall, f1, probs] = ttaPredict(net, N)
% TTAPREDICT  Test-time augmentation (TTA) on the CIFAR-10 test set.
%   Runs the trained network over the clean test images PLUS N randomly
%   augmented variations of them (same ±4 crop + h-flip used in training),
%   averages the softmax probabilities across all views, and reports the
%   accuracy + macro precision/recall/F1. Averaging over variations washes
%   out per-view quirks, so it usually buys a small accuracy bump.
%
% INPUTS:
%   net  - trained dlnetwork ending in softmax (from the trainnet engine)
%   N    - number of random variations to average over, on top of the clean
%          original. Total views = N + 1. N = 0 -> plain test accuracy.
%
% OUTPUTS:
%   acc       - overall accuracy (%)
%   precision - macro-averaged precision (%)
%   recall    - macro-averaged recall (%)
%   f1        - macro-averaged F1 score (%)
%   probs     - numClasses x numTest averaged softmax probabilities

    % ------------------ Handle optional arguments ------------------
    if nargin < 2 || isempty(N), N = 8; end


%% Loading the CIFAR-10 test data
% Same prep as loadtrainingdata: reshape, permute (CIFAR stores row/col/channel),
% and turn the labels into categoricals so the class order matches the network.

    dataDir = 'cifar-10-batches-mat';

    testBatch  = load(fullfile(dataDir, 'test_batch.mat'));
    testImages = reshape(testBatch.data',32,32,3,[]);
    testLabels = categorical(testBatch.labels);
    testImages = permute(testImages,[2 1 3 4]);

    cn      = categories(testLabels);
    numCls  = numel(cn);
    numTest = numel(testLabels);

%% Setting up the augmenter
% Exactly the train-time geometry: ±4 px shift + horizontal flip. Re-reading the
% datastore re-randomises it, so every pass is a fresh variation of the test set.

    imageAugmenter = imageDataAugmenter( ...
        'RandXTranslation',[-4 4], ...
        'RandYTranslation',[-4 4], ...
        'RandXReflection',true);

%% Scoring the clean original view

    cleanDS = augmentedImageDatastore([32 32 3], testImages, testLabels);
    probsSum = predictAllProbs(net, cleanDS, numCls, numTest);

%% Looping through the random variations
% Each pass reads a freshly augmented copy of the whole test set and adds its
% probabilities into the running sum.

    varDS = augmentedImageDatastore([32 32 3], testImages, testLabels, ...
        'DataAugmentation', imageAugmenter);

    for n = 1:N
        probsSum = probsSum + predictAllProbs(net, varDS, numCls, numTest);
    end

%% Averaging the views and predicting

    probs = probsSum / (N + 1);                 % numClasses x numTest

    [~, idx] = max(probs, [], 1);
    pred = categorical(idx(:)-1, 0:numCls-1, cn);

%% Confusion matrix and macro-averaged metrics
% Same per-class precision/recall/F1 as evaluateClassification, computed from
% the TTA-averaged predictions, then macro-averaged across the 10 classes.

    cm  = confusionmat(testLabels(:), pred);
    acc = mean(pred == testLabels(:)) * 100;

    numClasses = size(cm,1);
    prec = zeros(numClasses,1);
    rec  = zeros(numClasses,1);
    fsc  = zeros(numClasses,1);

    for c = 1:numClasses
        tp = cm(c,c);
        fp = sum(cm(:,c)) - tp;
        fn = sum(cm(c,:)) - tp;
        prec(c) = tp / (tp + fp + eps);
        rec(c)  = tp / (tp + fn + eps);
        fsc(c)  = 2 * (prec(c)*rec(c)) / (prec(c)+rec(c)+eps);
    end

    precision = mean(prec) * 100;
    recall    = mean(rec)  * 100;
    f1        = mean(fsc)  * 100;

    fprintf('\n=== TTA (%d views): acc %.2f%% | P %.2f%% | R %.2f%% | F1 %.2f%% ===\n', ...
        N+1, acc, precision, recall, f1);

end

% ------------------ Batched prediction over a datastore ------------------
function P = predictAllProbs(net, ds, numCls, numObs)
% Walks the datastore in 1000-image batches (keeps memory sane on 10k images)
% and returns the K x N softmax probabilities in the datastore's read order.

    ds.MiniBatchSize = 1000;
    reset(ds);

    P   = zeros(numCls, numObs, 'single');
    pos = 1;
    while hasdata(ds)
        b   = read(ds);
        X   = single(cat(4, b.input{:}));
        Y   = predict(net, dlarray(X,'SSCB'));
        Y   = gather(extractdata(Y));           % K x b
        nb  = size(Y,2);
        P(:, pos:pos+nb-1) = Y;
        pos = pos + nb;
    end
end
