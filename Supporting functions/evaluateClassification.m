% Compute confusion matrix, per-class metrics, and macro averages.
% INPUTS: net - trained network (DAGNetwork from trainNetwork, OR dlnetwork from trainnet) labels - ground-truth labels (categorical) DataSet - datastore to evaluate on outputFolder - folder to save results (if saveFlag is true) baseFilename - base filename for text file saveFlag - logical, true to save results to text file NAME - tag used in the results filename
% OUTPUTS: data = [acc macroPrecision macroRecall macroF1], and the confusion matrix cm.
% The engine is auto-detected: a dlnetwork is scored with predict (+ arg-max), a DAGNetwork with classify -- so this works for both training paths.
function [data,cm] = evaluateClassification(net,labels, DataSet, outputFolder, baseFilename, saveFlag,NAME)

if nargin < 5 || isempty(saveFlag)
    saveFlag = false;
end

% --- Predict (auto-detect dlnetwork vs DAGNetwork) ---
if isa(net,'dlnetwork')
    cn = categories(labels);
    reset(DataSet);
    Pred = categorical.empty(0,1);
    while hasdata(DataSet)
        b = read(DataSet);
        if istable(b), imgs = b{:,1}; else, imgs = b(:,1); end
        X = single(cat(4, imgs{:}));
        P = predict(net, dlarray(X,'SSCB'));
        [~,idx] = max(extractdata(gather(P)),[],1);
        Pred = [Pred; categorical(idx(:)-1, 0:numel(cn)-1, cn)]; %#ok<AGROW>
    end
else
    Pred = classify(net, DataSet);
end

% --- Confusion matrix and overall accuracy ---
cm = confusionmat(labels, Pred);
acc = mean(Pred(:) == labels(:));

% --- Number of classes and initialization ---
numClasses = size(cm,1);
precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1 = zeros(numClasses,1);

% --- Per-class metrics ---
for c = 1:numClasses
    tp = cm(c,c);
    fp = sum(cm(:,c)) - tp;
    fn = sum(cm(c,:)) - tp;

    precision(c) = tp / (tp + fp + eps);
    recall(c)    = tp / (tp + fn + eps);
    f1(c)        = 2 * (precision(c)*recall(c)) / (precision(c)+recall(c)+eps);
end

% --- Macro averages ---
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1);

% --- Save to text file if requested ---
if saveFlag
    resultsFile = fullfile(outputFolder, sprintf('results_%s.txt', NAME));
    fid = fopen(resultsFile,'w');
    fprintf(fid,'Model\t: %s\n',baseFilename);
    fprintf(fid,'Accuracy\t: %2.2f%%\n', acc*100);
    fprintf(fid,'Precision\t: %2.2f%%\n', macroPrecision*100);
    fprintf(fid,'Recall\t\t: %2.2f%%\n', macroRecall*100);
    fprintf(fid,'F1 Score\t: %2.2f%%\n', macroF1*100);
    fprintf(fid,'\nClass-wise metrics:\n');
    for c = 1:numClasses
        fprintf(fid,'Class %d ->\tPrecision: %2.2f%%\t, Recall: %2.2f%%\t, F1: %2.2f%%\n', ...
            c-1, precision(c)*100, recall(c)*100, f1(c)*100);
    end
    fclose(fid);
end
data = [acc macroPrecision macroRecall macroF1];

end
