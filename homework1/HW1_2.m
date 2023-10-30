load('TrainingSamplesDCT_8.mat');
freqFG = zeros(1, 64);
freqBG = zeros(1, 64);

% pick position of 2nd largest magnitude as the feature value
for row = 1 : length(TrainsampleDCT_FG)
    flatBlock = TrainsampleDCT_FG (row, :);
    [secondLargeNum, secondLargeIdx] = maxk(flatBlock, 2);
    freqFG(1, secondLargeIdx(2)) = freqFG(1, secondLargeIdx(2)) + 1;
end

% compute probability;
freqFG = freqFG / length(TrainsampleDCT_FG);

% plot histogramFG;
bar(linspace(1, 64, 64), freqFG);
xlabel('Pos');
ylabel('Probability')
savefig('histogramFG.fig');

% pick position of 2nd largest magnitude as the feature value
for row = 1 : length(TrainsampleDCT_BG)
    flatBlock = TrainsampleDCT_BG(row, :);
    [secondLargeNum, secondLargeIdx] = maxk(flatBlock, 2);
    freqBG(1, secondLargeIdx(2)) = freqBG(1, secondLargeIdx(2)) + 1;
end

% compute probability;
freqBG = freqBG / length(TrainsampleDCT_BG);

% plot histogramBG;
bar(linspace(1, 64, 64), freqBG);
xlabel('Pos');
ylabel('Probability')
savefig('histogramBG.fig');