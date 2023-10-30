all = zeros(255, 270);
allErrorRate2 = zeros(1, length(alpha));

fMean = mean(D1_FG, 1);
bMean = mean(D1_BG, 1);
fCov = cov(D1_FG);
bCov = cov(D1_BG);
fCovInv = inv(fCov);
bCovInv = inv(bCov);
fCovDet = det(fCov);
bCovDet = det(bCov);

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
        fRes = -0.5 * (flatBlock - fMean) * fCovInv * transpose(flatBlock - fMean) - ... 
        0.5 * log((2 * pi) ^ 64 * fCovDet) + log(priorFG);

        bRes = -0.5 * (flatBlock - bMean) * bCovInv * transpose(flatBlock - bMean) - ... 
        0.5 * log((2 * pi) ^ 64 * bCovDet) + log(priorBG);

        % Create a binary mask;
        if fRes >= bRes
            all(row, col) = 1;
        else
            all(row, col) = 0;
        end
    end
end

errorCount = 0;
sizes = size(mask);
rows = sizes(1);
cols = sizes(2);

% Check whether the predicted label equals to the ground truth label.
for row  = 1 : rows
    for col  = 1 : cols
        if (mask(row, col) / 255 ~= all(row, col))
            errorCount = errorCount + 1;
        end
    end
end

imshow(uint8(all), [0 1]);
savefig('Pred_2.fig');
clf;

% Compute error rate.
errorRate = errorCount / (rows * cols);
disp(errorRate);

% Plot the error rate curve.
x = zeros(1, length(alpha));
 
for i = 1 : length(alpha)
    x(1, i) = log(alpha(1, i));
    allErrorRate2(1, i) = errorRate;
end

plot(x, allErrorRate2);
savefig('allErrorRate_2.fig');
clf;