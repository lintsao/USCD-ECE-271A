% Read img.
img = imread('hw4Data/cheetah.bmp');
img = double(img) / 255;
mask = imread('hw4Data/cheetah_mask.bmp');
 
% Padding img with zero.
I = zeros(263, 278);
for row = 5 : 259
    for col = 5 : 274
        I(row, col) = img(row - 4, col - 4);
    end
end

% Read pattern.
pattern = readmatrix('hw4Data/Zig-Zag Pattern.txt');

load('hw4Data/TrainingSamplesDCT_8_new.mat');

% Hyper parameters.
M = 5;
C = 8;
fgSize = length(TrainsampleDCT_FG);
bgSize = length(TrainsampleDCT_BG);
maxIter = 200;

dim = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];
errorList = zeros(5, 5, length(dim));

% Pre-compute DCT features for all image blocks
rows = 255;
cols = 270;
im_blocks = zeros(rows * cols, 64);
idx = 1;
for r0 = 1 : rows
    for c0 = 1 : cols
        block = zeros(8, 8);
        for r = r0 : r0 + 7
            for c = c0 : c0 + 7
                block(r - r0 + 1, c - c0 + 1) = I(r, c);
            end
        end
        dct2Block = dct2(block);
        flatBlock = zeros(1, 64);
        for r = 1 : 8
            for c = 1 : 8
                flatBlock(1, pattern(r, c) + 1) = dct2Block(r, c);
            end
        end
        im_blocks(idx, :) = flatBlock;
        idx = idx + 1;
    end
end

for idxFG = 1 : 5
    piFG = ones(1, C) / C;
    muFG = rand(C, 64);
    covFG = zeros(C, 64, 64);

    for i = 1 : C
        covTmp = normrnd(5, 0.3,[1, 64]);
        covFG(i, :, :) = diag(covTmp);
    end

    % EM foreground.
    disp("idxFG " + idxFG);
    for m = 1 : maxIter
        % E-step.

        H = zeros(C, fgSize);

        for i = 1 : C
            H(i, :) = mvnpdf(TrainsampleDCT_FG, muFG(i, :), squeeze(covFG(i, :, :))) * piFG(1, i);
        end

        H = transpose(H);
        H = H ./sum(H, 2);

%         for j = 1 : fgSize
%             for c = 1 : C
%                 H(j, c) = H(j, c) / Hsum2(j, 1);
%             end
%         end

        Hsum1 = sum(H, 1);

        % M-step.
        % update pi.
        piFG = 1 / fgSize * Hsum1;

        % update mu.
        muNew = zeros(C, 64);
       
        for i = 1 : C
            HTmp = TrainsampleDCT_FG;
            for j = 1 : fgSize
                for k = 1 : 64
                    HTmp(j, k) = HTmp(j, k) * H(j, i);
                end
            end

            muNew(i, :, :) = sum(HTmp, 1) / Hsum1(1, i);
        end

        % update cov.
        covNew = zeros(C, 64, 64);
        for i = 1 : C
            xTmp = TrainsampleDCT_FG - muFG(i, :);
            for j = 1 : fgSize
                for k = 1 : 64
                    xTmp(j, k) = xTmp(j, k) * xTmp(j, k) * H(j, i);
                end
            end

            covTmp = sum(xTmp, 1) / Hsum1(1, i);

            for j = 1 : 64
                if covTmp(1, j) < 0.000001
                    covTmp(1, j) = 0.000001;
                end
            end

            covNew(i, :, :) = diag(covTmp);
        end
        
        muFG = muNew;
        covFG = covNew;
    end

    for idxBG = 1 : 5
        disp("idxBG " + idxBG);
        piBG = ones(1, C) / C;
        muBG = rand(C, 64);
        covBG = zeros(C, 64, 64);
        for m = 1 : maxIter
            % E-step.
    
            H = zeros(C, bgSize);
    
            for i = 1 : C
            H(i, :) = mvnpdf(TrainsampleDCT_BG, muBG(i, :), squeeze(covBG(i, :, :))) * piBG(1, i);
            end
    
            H = transpose(H);
            Hsum2 = sum(H, 2);
    
            for j = 1 : bgSize
                for c = 1 : C
                    H(j, c) = H(j, c) / Hsum2(j, 1);
                end
            end
    
            Hsum1 = sum(H, 1);
    
            % M-step.
            % update pi.
            piBG = 1 / bgSize * Hsum1;
    
            % update mu.
            muNew = zeros(C, 64);
           
            for i = 1 : C
                HTmp = TrainsampleDCT_BG;
                for j = 1 : bgSize
                    for k = 1 : 64
                        HTmp(j, k) = HTmp(j, k) * H(j, i);
                    end
                end
    
                muNew(i, :, :) = sum(HTmp, 1) / Hsum1(1, i);
            end
    
            % update cov.
            covNew = zeros(C, 64, 64);
            for i = 1 : C
                xTmp = TrainsampleDCT_BG - muBG(i, :);
                for j = 1 : bgSize
                    for k = 1 : 64
                        xTmp(j, k) = xTmp(j, k) * xTmp(j, k) * H(j, i);
                    end
                end
    
                covTmp = sum(xTmp, 1) / Hsum1(1, i);
    
                for j = 1 : 64
                    if covTmp(1, j) < 0.000001
                        covTmp(1, j) = 0.000001;
                    end
                end
    
                covNew(i, :, :) = diag(covTmp);
            end
            
            muBG = muNew;
            covBG = covNew;
        end

        for d = 1 : length(dim)
            currD = dim(1, d);
            disp("currD " + currD);


            muFGCur = muFG(:, 1 : currD);
            covFGCur = covFG(:, 1 : currD, 1 : currD);
            muBGCur = muBG(:, 1 : currD);
            covBGCur = covBG(:, 1 : currD, 1 : currD);

            % Use pre-computed DCT features stored in im_blocks
            blocksCur = im_blocks(:, 1 : currD);

            probFG = zeros(rows * cols, 1);
            probBG = zeros(rows * cols, 1);

            for k = 1 : C
                probFG = probFG + mvnpdf(blocksCur, muFGCur(k, :), squeeze(covFGCur(k, :, :))) * piFG(1, k);
            end

            for k = 1 : C
                probBG = probBG + mvnpdf(blocksCur, muBGCur(k, :), squeeze(covBGCur(k, :, :))) * piBG(1, k);
            end

            A = reshape(probFG - probBG, rows, cols);

            for row = 1 : rows
                for col = 1 : cols
                    if A(row, col) >= 0
                        A(row, col) = 1;
                    else
                        A(row, col) = 0;
                    end

                    if mask(row, col) / 255 ~= A(row, col)
                        errorList(idxFG, idxBG, currD) = errorList(idxFG, idxBG, currD) + 1;
                    end
                end
            end

            errorList(idxFG, idxBG, currD) = errorList(idxFG, idxBG, currD) / (rows * cols);
        end
    end
end

% Plot the gram.
for idxFG = 1 : 5
    subplot(1, 5, idxFG);
    plot(dim, errorList(idxFG, 1, :), dim, errorList(idxFG, 2, :), dim, errorList(idxFG, 3, :), dim, errorList(idxFG, 4, :), dim, errorList(idxFG, 5, :), '.-'), legend('BG1', 'BG2', 'BG3', 'BG4', 'BG5');
    title(['error ', idxFG]);
    xlabel('dim');
    ylabel('error rate') ;
end
