% read img;
img = imread('cheetah.bmp');
img = double(img) / 255;

% padding img with zero;
I = zeros(263, 278);
for row = 5 : 259
    for col = 5 : 274
        I(row, col) = img(row - 4, col - 4);
    end
end

% read pattern;
pattern = readmatrix('Zig-Zag Pattern.txt');

% create the bitmask by all features;
all = zeros(255, 270);

fCov = cov(TrainsampleDCT_FG);
bCov = cov(TrainsampleDCT_BG);
fCovInv = inv(fCov);
bCovInv = inv(bCov);
fCovDet = det(fCov);
bCovDet = det(bCov);
fAllMean = mean(TrainsampleDCT_FG, 1);
bAllMean = mean(TrainsampleDCT_BG, 1);

for row = 1 : 255
    for col = 1 : 270
        block = zeros(8, 8);
        % get the blcok;
        for r = row : row + 7
            for c = col : col + 7
                block(r - row + 1, c - col + 1) = I(r, c);
            end
        end

        % compute DCT;
        dct2Block = dct2(block);
        flatBlock = zeros(1, 64);
        
        % order coefficients with zig-zag scan;
        for r = 1 : 8
            for c = 1 : 8
                flatBlock(1, pattern(r, c) + 1) = dct2Block(r, c);
            end
        end
        
        % use Gaussian classifier to find class Y for each block;
        fRes = -0.5 * (flatBlock - fAllMean) * fCovInv * transpose(flatBlock - fAllMean) - ... 
        0.5 * log((2 * pi) ^ 64 * fCovDet) + log(priorFG);

        bRes = -0.5 * (flatBlock - bAllMean) * bCovInv * transpose(flatBlock - bAllMean) - ... 
        0.5 * log((2 * pi) ^ 64 * bCovDet) + log(priorBG);

        % create a binary mask;
        if fRes >= bRes
            all(row, col) = 1;
        else
            all(row, col) = 0;
        end
    end
end

imshow(uint8(all), [0 1]);
savefig('all.fig');

% create the bitmask by best 8 features;
best8 = zeros(255, 270);

fCov = cov(TrainsampleDCT_FG(:, best));
bCov = cov(TrainsampleDCT_BG(:, best));
fCovInv = inv(fCov);
bCovInv = inv(bCov);
fCovDet = det(fCov);
bCovDet = det(bCov);
fBestMean = mean(TrainsampleDCT_FG(:, best), 1);
bBestMean = mean(TrainsampleDCT_BG(:, best), 1);

for row = 1 : 255
    for col = 1 : 270
        block = zeros(8, 8);
        % get the blcok;
        for r = row : row + 7
            for c = col : col + 7
                block(r - row + 1, c - col + 1) = I(r, c);
            end
        end

        % compute DCT;
        dct2Block = dct2(block);
        flatBlock = zeros(1, 64);
        
        % order coefficients with zig-zag scan;
        for r = 1 : 8
            for c = 1 : 8
                flatBlock(1, pattern(r, c) + 1) = dct2Block(r, c);
            end
        end
        flatBlock = flatBlock(best);
        
        % use Gaussian classifier to find class Y for each block;
        fRes = -0.5 * (flatBlock - fBestMean)  * fCovInv * transpose(flatBlock - fBestMean) - ... 
        0.5 * log((2 * pi) ^ 64 * fCovDet) + log(priorFG);

        bRes = -0.5  * (flatBlock - bBestMean) * bCovInv * transpose(flatBlock - bBestMean) - ... 
        0.5 * log((2 * pi) ^ 64 * bCovDet) + log(priorBG);

        % create a binary mask;
        if fRes >= bRes
            best8(row, col) = 1;
        else
            best8(row, col) = 0;
        end
    end
end

imshow(uint8(best8), [0 1]);
savefig('best8.fig');

% compute the error rate;https://www.pramp.com/invt/WnZA1lq2boIGngN71LAY
mask = imread('cheetah_mask.bmp');
allErrorCount = 0;
best8ErrorCount = 0;
sizes = size(mask);
rows = sizes(1);
cols = sizes(2);

% check whether the predicted label equals to the ground truth label;
for row  = 1 : rows
    for col  = 1 : cols
        if (mask(row, col) / 255 ~= all(row, col))
            allErrorCount = allErrorCount + 1;
        end
        if (mask(row, col) / 255 ~= best8(row, col))
            best8ErrorCount = best8ErrorCount + 1;
        end
    end
end

allErrorRate = allErrorCount / (rows * cols);
disp(allErrorRate);
best8ErrorRate = best8ErrorCount / (rows * cols);
disp(best8ErrorRate);