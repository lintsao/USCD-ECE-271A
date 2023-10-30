% Compute the MLE for the foreground;
fSize = size(TrainsampleDCT_FG);
fSize = fSize(1);
fBase = zeros(1, 64);
fTmp = zeros(1, 64);
fMean = mean(TrainsampleDCT_FG, 1);
fVar = var(TrainsampleDCT_FG, 1);
fStd = std(TrainsampleDCT_FG, 1);

for col = 1 : 64
    fBase(col) = (-2/fSize) * log (2 * pi);
end

for col = 1 : 64
    for row = 1 : fSize
        fTmp(col) = fTmp(col) + ((TrainsampleDCT_FG(row, col) - fMean(col)) / fStd(col))^2;
    end

    fTmp(col) = 0.5 * fTmp(col);
end

fMLELog = fBase - fSize * log(fStd) - fTmp;

% Compute the MLE for the background;
bSize = size(TrainsampleDCT_BG);
bSize = bSize(1);
bBase = zeros(1, 64);
bTmp = zeros(1, 64);
bMean = mean(TrainsampleDCT_BG, 1);
bVar = var(TrainsampleDCT_BG, 1);
bStd = std(TrainsampleDCT_BG, 1);

for col = 1 : 64
    bBase(col) = (-2/bSize) * log (2 * pi);
end

for col = 1 : 64
    for row = 1 : bSize
        bTmp(col) = bTmp(col) + ((TrainsampleDCT_BG(row, col) - bMean(col)) / bStd(col))^2;
    end

    bTmp(col) = 0.5 * bTmp(col);
end

bMLELog = bBase - bSize * log(bStd) - bTmp;

% plot the image;
sampleNum = 100;
stdNum = 5;
distance = zeros(1, 64);

for i = 1 : 64
    subplot(8, 8, i);
    fX = linspace(fMean(i) - stdNum * fStd(i), fMean(i) + stdNum * fStd(i), sampleNum);
    bX = linspace(bMean(i) - stdNum * bStd(i), bMean(i) + stdNum * bStd(i), sampleNum);
    x = sort([fX, bX]);

    fY = normpdf(x, fMean(i), fStd(i));
    bY = normpdf(x, bMean(i), bStd(i));
    distance(i) = abs(fMean(i) - bMean(i));

    plot(x, fY, x, bY);
end
savefig('all features.fig');

% plot the best and the worst feature;
[out, idx] = sort(distance, "descend");
best = idx(1 : 8);
worst = fliplr(idx(57 : 64));

% best;
for i = 1 : size(best, 2)
    subplot(2, 4, i);
    fX = linspace(fMean(best(i)) - stdNum * fStd(best(i)), fMean(best(i)) + stdNum * fStd(best(i)), sampleNum);
    bX = linspace(bMean(best(i)) - stdNum * bStd(best(i)), bMean(best(i)) + stdNum * bStd(best(i)), sampleNum);
    x = sort([fX, bX]);

    fY = normpdf(x, fMean(best(i)), fStd(best(i)));
    bY = normpdf(x, bMean(best(i)), bStd(best(i)));

    plot(x, fY, x, bY);
    title(['best ', num2str(i)]);
end
savefig('best 8 features.fig');

% worst;
for i = 1 : size(worst, 2)
    subplot(2, 4, i);
    fX = linspace(fMean(worst(i)) - stdNum * fStd(worst(i)), fMean(worst(i)) + stdNum * fStd(worst(i)), sampleNum);
    bX = linspace(bMean(worst(i)) - stdNum * bStd(worst(i)), bMean(worst(i)) + stdNum * bStd(worst(i)), sampleNum);
    x = sort([fX, bX]);

    fY = normpdf(x, fMean(worst(i)), fStd(worst(i)));
    bY = normpdf(x, bMean(worst(i)), bStd(worst(i)));

    plot(x, fY, x, bY);
    title(['wosrt ', num2str(i)]);
end
savefig('worst 8 features.fig');