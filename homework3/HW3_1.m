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

% Read pattern;
pattern = readmatrix('hw3Data/Zig-Zag Pattern.txt');

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

% Compute covariance.
fSize = size(D1_FG);
bSize = size(D1_BG);
fCov = cov(D1_FG);
bCov = cov(D1_BG);
fMean = mean(D1_FG, 1);
bMean = mean(D1_BG, 1);

fSize = size(D1_FG);
bSize = size(D1_BG);
fCov = cov(D1_FG);
bCov = cov(D1_BG);
fMean = mean(D1_FG, 1);
bMean = mean(D1_BG, 1);

% Compute cov0.
cov01 = zeros(length(alpha), 64, 64);
for i = 1 : length(alpha)
    for row = 1 : 64
        for col = 1 : 64
            if row == col
                cov01(i, row, col) = alpha(i) * w01(1, row);
            end
        end
    end
end

% Compute mu1, cov1.
totalMu1FG = zeros(length(alpha), 64, 1);
totalMu1BG = zeros(length(alpha), 64, 1);
totalCov1FG = zeros(length(alpha), 64, 64);
totalCov1BG = zeros(length(alpha), 64, 64);

for i = 1 : length(alpha)
   tmp = squeeze(cov01(i, :, :));
   mu1FG = tmp * inv(tmp + fCov / fSize(1)) * transpose(fMean) + ...
       1 / fSize(1) * fCov * inv(tmp + fCov / fSize(1)) * mu0FG1;
   mu1BG = tmp * inv(tmp + bCov / bSize(1)) * transpose(bMean) + ...
       1 / bSize(1) * bCov * inv(tmp + bCov / bSize(1)) * mu0BG1;
    
   cov1FG = tmp * inv(tmp + fCov / fSize(1)) * fCov / fSize(1);
   cov1BG = tmp * inv(tmp + bCov / bSize(1)) * bCov / bSize(1);

   totalMu1FG(i, :, :) = mu1FG;
   totalMu1BG(i, :, :) = mu1BG;
   totalCov1FG(i, :, :) = cov1FG;
   totalCov1BG(i, :, :) = cov1BG;
end

% Compute the predictive distribution.
totalCovPd1FG = zeros(length(alpha), 64, 64);
totalCovPd1BG = zeros(length(alpha), 64, 64);

for i = 1 : length(alpha)
    for row = 1 : 64
        for col = 1 : 64
            totalCovPd1FG(i, row, col) = totalCov1FG(i, row, col) + fCov(row, col);
            totalCovPd1BG(i, row, col) = totalCov1BG(i, row, col) + bCov(row, col);
        end
    end
end

% Compute prior.
priorFG = fSize / (fSize + bSize);
priorBG = bSize / (fSize + bSize);

% Predictive distribution.
all = zeros(length(alpha), 255, 270);
allErrorRate1 = zeros(1, length(alpha));

for i = 1 : length(alpha)
    covPd1FGInv = inv(squeeze(totalCovPd1FG(i, :, :)));
    covPd1BGInv = inv(squeeze(totalCovPd1BG(i, :, :)));
    covPd1FGDet = det(squeeze(totalCovPd1FG(i, :, :)));
    covPd1BGDet = det(squeeze(totalCovPd1BG(i, :, :)));

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
            
            % Use BDR to find class Y for each block.
            fRes = -0.5 * (flatBlock - totalMu1FG(i, :)) * covPd1FGInv * transpose(flatBlock - totalMu1FG(i, :)) - ... 
            0.5 * log((2 * pi) ^ 64 * covPd1FGDet) + log(priorFG);
    
            bRes = -0.5 * (flatBlock - totalMu1BG(i, :)) * covPd1BGInv * transpose(flatBlock - totalMu1BG(i, :)) - ... 
            0.5 * log((2 * pi) ^ 64 * covPd1BGDet) + log(priorBG);
    
            % Create a binary mask;
            if fRes >= bRes
                all(i, row, col) = 1;
            else
                all(i, row, col) = 0;
            end
        end
    end

    errorCount = 0;
    sizes = size(mask);
    rows = sizes(1);
    cols = sizes(2);
    
    % Check whether the predicted label equals to the ground truth label;
    for row  = 1 : rows
        for col  = 1 : cols
            if (mask(row, col) / 255 ~= all(i, row, col))
                errorCount = errorCount + 1;
            end
        end
    end 
    
    % Compute error rate;
    allErrorRate1(1, i) = errorCount / (rows * cols);
    disp(allErrorRate1(1, i));
    
    % Plot the image.
    subplot(3, 3, i);
    subimage(uint8(squeeze(all(i, :, :))), [0 1]);
end
savefig('Pred_1.fig');
clf;

% Plot the error rate curve.
x = zeros(1, length(alpha));
 
for i = 1 : length(alpha)
    x(1, i) = log(alpha(1, i));
end

plot(x, allErrorRate1);
savefig('allErrorRate_1.fig');
clf;