function dataOut = cutoutAugment(dataIn)
%CUTOUTAUGMENT Apply random cutout (random erasing) to a minibatch.
%   Used as a transform() on an augmentedImageDatastore. Each image has a
%   single square patch (16x16) erased to zero with probability 0.5, at a
%   random location. This is the DeVries & Taylor cutout regularizer, which
%   reduces overfitting on CIFAR-10 (helps fine-grained classes like cat/dog).
%
%   dataIn is the table returned by augmentedImageDatastore read, with image
%   data in the first variable ('input') and labels in the second
%   ('response'). The label column is passed through unchanged.

    dataOut = dataIn;

    if istable(dataIn)
        vn   = dataIn.Properties.VariableNames;
        imgs = dataIn.(vn{1});
        for i = 1:numel(imgs)
            imgs{i} = applyCutout(imgs{i});
        end
        dataOut.(vn{1}) = imgs;
    elseif iscell(dataIn) && size(dataIn,2) >= 1
        for i = 1:size(dataIn,1)
            dataOut{i,1} = applyCutout(dataIn{i,1});
        end
    else
        dataOut = applyCutout(dataIn);
    end
end

function img = applyCutout(img)
    if rand > 0.5
        return;                      % apply with probability 0.5
    end
    [h, w, ~] = size(img);
    s  = 16;                         % patch size (pixels)
    cy = randi(h);  cx = randi(w);   % random center
    y1 = max(1, cy - floor(s/2));  y2 = min(h, y1 + s - 1);
    x1 = max(1, cx - floor(s/2));  x2 = min(w, x1 + s - 1);
    img(y1:y2, x1:x2, :) = 0;        % erase patch to zero
end
