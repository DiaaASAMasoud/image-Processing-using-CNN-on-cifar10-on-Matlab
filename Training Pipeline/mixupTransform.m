function out = mixupTransform(data, numClasses, alpha, smooth, classNames)
%MIXUPTRANSFORM Mixup + label smoothing for a minibatch (Step 3, trainnet).
%   Used as a transform() on an augmentedImageDatastore for the final Step 3
%   training. Converts a batch of images+labels into mixed images and soft
%   targets:
%     - label smoothing: T = (1-smooth)*onehot + smooth/numClasses
%     - mixup: blend each image with a shuffled partner by lambda~Beta(a,a),
%       and blend their soft targets the same way.
%   Returns a B-by-2 cell, one row per observation {image, softTarget}, so
%   trainnet collates it into SSCB/CB itself. Mixup is applied across the
%   batch before the rows are split out.
%   Validation data is left clean (call with alpha=0, smooth=0 -> just one-hot).

    if istable(data)
        imgs = data{:,1};
        labs = data{:,2};
    else
        imgs = data(:,1);
        labs = data(:,2);
    end
    B = numel(imgs);

    X = single(cat(4, imgs{:}));                       % H x W x C x B

    labs = categorical(labs, classNames);
    T = single(onehotencode(labs(:).', 1));            % numClasses x B
    T = (1 - smooth) * T + smooth / numClasses;        % label smoothing

    if alpha > 0
        lam  = betarnd(alpha, alpha);
        perm = randperm(B);
        X = lam * X + (1 - lam) * X(:,:,:,perm);
        T = lam * T + (1 - lam) * T(:,perm);
    end

    out = cell(B, 2);
    for i = 1:B
        out{i,1} = X(:,:,:,i);                          % H x W x C
        out{i,2} = T(:,i);                              % numClasses x 1
    end
end
