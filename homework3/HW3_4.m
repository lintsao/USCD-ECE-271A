%%%%%%%%%% Load data. %%%%%%%%%%

% Read img.
img = imread('hw3Data/cheetah.bmp');
img = double(img) / 255;
mask = imread('cheetah_mask.bmp');
 
% Padding img with zero.
I = zeros(263, 278);
for row = 5 : 259
    for col = 5 : 274
        I(row, col) = img(row - 4, col - 4);
    end
end

% Read pattern.
pattern = readmatrix('hw3Data/Zig-Zag Pattern.txt');

% Read training data.
load('hw3Data/TrainingSamplesDCT_subsets_8.mat');

load('hw3Data/Prior_1.mat');
w01 = W0;
mu0FG1 = transpose(mu0_FG);
mu0BG1 = transpose(mu0_BG);

load('hw3Data/Prior_2.mat')
w02 = W0;
mu0FG2 = transpose(mu0_FG);
mu0BG2 = transpose(mu0_BG);

clear W0;
clear mu0_FG;
clear mu0_BG;

load('hw3Data/Alpha.mat')

% Choose S.
for s = 1 : 2
    if s == 1
        wS = w01;
        muSFG = mu0FG1;
        muSBG = mu0BG1;
    else
        wS = w02;
        muSFG = mu0FG2;
        muSBG = mu0BG2;
    end

    % Choose D.
    for d = 1 : 4
        if d == 1
            DFG = D1_FG;
            DBG = D1_BG;

        elseif d == 2
            DFG = D2_FG;
            DBG = D2_BG;

        elseif d == 3
            DFG = D3_FG;
            DBG = D3_BG;

        else
            DFG = D4_FG;
            DBG = D4_BG;
        end

        % Compute covariance.
        fSize = size(DFG);
        bSize = size(DBG);
        fCov = cov(DFG);
        bCov = cov(DBG);
        fMean = mean(DFG, 1);
        bMean = mean(DBG, 1);
        fCovInv = inv(fCov);
        bCovInv = inv(bCov);
        fCovDet = det(fCov);
        bCovDet = det(bCov);
        
        % Compute cov0.
        cov0 = zeros(length(alpha), 64, 64);
        for i = 1 : length(alpha)
            for row = 1 : 64
                for col = 1 : 64
                    if row == col
                        cov0(i, row, col) = alpha(i) * wS(1, row);
                    end
                end
            end
        end
        
        % Compute mu1, cov1.
        totalMuFG = zeros(length(alpha), 64, 1);
        totalMuBG = zeros(length(alpha), 64, 1);
        totalCovFG = zeros(length(alpha), 64, 64);
        totalCovBG = zeros(length(alpha), 64, 64);
        
        for i = 1 : length(alpha)
           tmp = squeeze(cov0(i, :, :));
           muFG = tmp * inv(tmp + fCov / fSize(1)) * transpose(fMean) + ...
               1 / fSize(1) * fCov * inv(tmp + fCov / fSize(1)) * muSFG;
           muBG = tmp * inv(tmp + bCov / bSize(1)) * transpose(bMean) + ...
               1 / bSize(1) * bCov * inv(tmp + bCov / bSize(1)) * muSBG;
            
           covFG = tmp * inv(tmp + fCov / fSize(1)) * fCov / fSize(1);
           covBG = tmp * inv(tmp + bCov / bSize(1)) * bCov / bSize(1);
        
           totalMuFG(i, :, :) = muFG;
           totalMuBG(i, :, :) = muBG;
           totalCovFG(i, :, :) = covFG;
           totalCovBG(i, :, :) = covBG;
        end
        
        % Compute the predictive distribution.
        totalCovPdFG = zeros(length(alpha), 64, 64);
        totalCovPdBG = zeros(length(alpha), 64, 64);
        
        for i = 1 : length(alpha)
            for row = 1 : 64
                for col = 1 : 64
                    totalCovPdFG(i, row, col) = totalCovFG(i, row, col) + ...
                        fCov(row, col);
                    totalCovPdBG(i, row, col) = totalCovBG(i, row, col) + ...
                        bCov(row, col);
                end
            end
        end
        
        % Compute prior.
        priorFG = fSize / (fSize + bSize);
        priorBG = bSize / (fSize + bSize);
        
        % Predict.
        all1 = zeros(length(alpha), 255, 270);
        all2 = zeros(length(alpha), 255, 270);
        all3 = zeros(length(alpha), 255, 270);

        allErrorRate1 = zeros(1, length(alpha));
        allErrorRate2 = zeros(1, length(alpha));
        allErrorRate3 = zeros(1, length(alpha));
        
        for i = 1 : length(alpha)
            covPdFGInv = inv(squeeze(totalCovPdFG(i, :, :)));
            covPdBGInv = inv(squeeze(totalCovPdBG(i, :, :)));
            covPdFGDet = det(squeeze(totalCovPdFG(i, :, :)));
            covPdBGDet = det(squeeze(totalCovPdBG(i, :, :)));
        
            for row = 1 : 255
                for col = 1 : 270
                    block = zeros(8, 8);
                    % Get the blcok.
                    for r = row : row + 7
                        for c = col : col + 7
                            block(r - row + 1, c - col + 1) = I(r, c);
                        end
                    end
            
                    % Compute DCT.
                    dct2Block = dct2(block);
                    flatBlock = zeros(1, 64);
        
                    for r = 1 : 8
                        for c = 1 : 8
                            flatBlock(1, pattern(r, c) + 1) = dct2Block(r, c);
                        end
                    end
                    
                    % PD: Use BDR to find class Y for each block.
                    fRes = -0.5 * (flatBlock - totalMuFG(i, :)) * covPdFGInv * ...
                        transpose(flatBlock - totalMuFG(i, :)) - ... 
                    0.5 * log((2 * pi) ^ 64 * covPdFGDet) + log(priorFG);
            
                    bRes = -0.5 * (flatBlock - totalMuBG(i, :)) * covPdBGInv * ...
                        transpose(flatBlock - totalMuBG(i, :)) - ... 
                    0.5 * log((2 * pi) ^ 64 * covPdBGDet) + log(priorBG);
            
                    % Create a binary mask;
                    if fRes >= bRes
                        all1(i, row, col) = 1;
                    else
                        all1(i, row, col) = 0;
                    end

                    % ML: Use BDR to find class Y for each block.
                    fRes = -0.5 * (flatBlock - fMean) * fCovInv * transpose(flatBlock - fMean) - ... 
                    0.5 * log((2 * pi) ^ 64 * fCovDet) + log(priorFG);
            
                    bRes = -0.5 * (flatBlock - bMean) * bCovInv * transpose(flatBlock - bMean) - ... 
                    0.5 * log((2 * pi) ^ 64 * bCovDet) + log(priorBG);

                    % Create a binary mask;
                    if fRes >= bRes
                        all2(i, row, col) = 1;
                    else
                        all2(i, row, col) = 0;
                    end

                    % Use BDR to find class Y for each block.
                    fRes = -0.5 * (flatBlock - totalMuFG(i, :)) * fCovInv * ...
                        transpose(flatBlock - totalMuFG(i, :)) - ...
                        0.5 * log((2 * pi) ^ 64 * fCovDet) + log(priorFG);
            
                    bRes = -0.5 * (flatBlock - totalMuBG(i, :)) * bCovInv * ...
                        transpose(flatBlock - totalMuBG(i, :)) - ... 
                        0.5 * log((2 * pi) ^ 64 * bCovDet) + log(priorBG);

                    % Create a binary mask;
                    if fRes >= bRes
                        all3(i, row, col) = 1;
                    else
                        all3(i, row, col) = 0;
                    end
                end
            end

            errorCount1 = 0;
            errorCount2 = 0;
            errorCount3 = 0;

            sizes = size(mask);
            rows = sizes(1);
            cols = sizes(2);
            
            % Check whether the predicted label equals to the ground truth label.
            for row  = 1 : rows
                for col  = 1 : cols
                    if (mask(row, col) / 255 ~= all1(i, row, col))
                        errorCount1 = errorCount1 + 1;
                    end

                    if (mask(row, col) / 255 ~= all2(i, row, col))
                        errorCount2 = errorCount2 + 1;
                    end

                    if (mask(row, col) / 255 ~= all3(i, row, col))
                        errorCount3 = errorCount3 + 1;
                    end
                end
            end

            allErrorRate1(1, i) = errorCount1 / (rows * cols);
            allErrorRate2(1, i) = errorCount2 / (rows * cols);
            allErrorRate3(1, i) = errorCount3 / (rows * cols);
            
            disp('%%%%%%%%%%');
            disp(allErrorRate1(1, i));
            disp(allErrorRate2(1, i));
            disp(allErrorRate3(1, i));
            disp(s);
            disp(d);
            disp(i);
            disp('%%%%%%%%%%');
        end

        % Plot the error rate curve.
        x = zeros(1, length(alpha));
         
        for i = 1 : length(alpha)
            x(1, i) = log(alpha(1, i));
        end

        % Compared with all curves.
        subplot(4, 2, (d - 1) * 2 + s)
        plot(x, allErrorRate1, x, allErrorRate2, x, allErrorRate3, '.-'), legend('PD', 'ML', 'MAP');
        title(['Strategy ', num2str(s), ' Dataset ', num2str(d)]);
        xlabel('log(alpha)');
        ylabel('error rate') ;
    end
end

savefig('allErrorRate_3_4.fig');
clf;