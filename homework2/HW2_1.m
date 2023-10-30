load('TrainingSamplesDCT_8_new.mat');
total = size(TrainsampleDCT_FG) + size(TrainsampleDCT_BG);
priorFG = length(TrainsampleDCT_FG) / total(1);
priorBG = length(TrainsampleDCT_BG) / total(1);