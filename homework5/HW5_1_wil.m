% load data
load('hw5Data/TrainingSamplesDCT_8_new.mat');

% load image, mask
image = imread('hw5Data/cheetah.bmp');
% image = double(image) / 255;
image = im2double(image);
mask = imread('hw5Data/cheetah_mask.bmp');
mask = im2double(mask);
zigzag = readmatrix('hw5Data/Zig-Zag Pattern.txt');

% base stats
BG_samples = size(TrainsampleDCT_BG, 1);
BG_dim = size(TrainsampleDCT_BG, 2);
FG_samples = size(TrainsampleDCT_FG, 1);
FG_dim = size(TrainsampleDCT_FG, 2);


% padding for further comparison
I = zeros(263, 278);
for i = 5:259
    for j = 5:274
        I(i,j) = image(i-4, j-4);
    end
end

% sliding window to get dct features
A = zeros(255*270, 64);
for i = 1:255
    for j = 1:270
        block = zeros(8,8);
        for row = i:i+7
            for col = j:j+7
                block(row-i+1, col-j+1) = I(row, col);
            end
        end
%       convert with dct
        bk2dct = dct2(block);
        x = zeros(1, 64);
        for a = 1:8
            for b = 1:8
%               order coefficients with zig-zag scan
                x(1, zigzag(a,b)+1) = bk2dct(a,b);
            end
        end
        A((i-1)*270+j,:) = x;
    end
end

% settings
C = 8;
dim_list = [1,2,4,8,16,24,32,40,48,56,64];
iterations = 200;

% fix BG
for a = 1:5
    % FG EM
    pi_FG = randi(1, C);
    pi_FG = pi_FG / sum(pi_FG);
    % random pick 8 feat
    mu_FG = TrainsampleDCT_FG(randi([1 FG_samples], 1, C), :);
    % randomize 8 diagnol sigma
    sigma_FG = zeros(FG_dim, FG_dim, C);
    for i = 1:C
        sigma_FG(:, :, i) = (rand(1, FG_dim).*eye(FG_dim));
    end
    
    joint = zeros(FG_samples, C);
    for i = 1:iterations
    %     E-step
        for j = 1:C
            joint(:, j) = mvnpdf(TrainsampleDCT_FG, mu_FG(j, :), sigma_FG(:, :, j)) * pi_FG(j);
        end
        hij = joint ./ sum(joint, 2);
        loglld(i) = sum(log(sum(joint,2)));
    
    %     M-step
    %     update
        pi_FG = sum(hij) / FG_samples;
        mu_FG = (hij' * TrainsampleDCT_FG) ./ sum(hij)';
        for j = 1:C
            sigma_FG(:,:,j) = diag(diag((TrainsampleDCT_FG - mu_FG(j, :))' ...
                .* hij(:, j)' * (TrainsampleDCT_FG - mu_FG(j,:)) ./ sum(hij(:, j),1))+0.000001);
        end
    
        if i > 1
            if abs(loglld(i) - loglld(i-1)) < 0.001
                break;
            end
        end
    end


    for b = 1:5
        % BG EM
        pi_BG = randi(1, C);
        pi_BG = pi_BG / sum(pi_BG);
        % random pick 8 feat
        mu_BG = TrainsampleDCT_BG(randi([1 BG_samples], 1, C), :);
        % randomize 8 diagnol sigma
        sigma_BG = zeros(BG_dim, BG_dim, C);
        for i = 1:C
            sigma_BG(:, :, i) = (rand(1, BG_dim).*eye(BG_dim));
        end
        
        joint = zeros(BG_samples, C);
        for i = 1:iterations
        %     E-step
            for j = 1:C
                joint(:, j) = mvnpdf(TrainsampleDCT_BG, mu_BG(j, :), sigma_BG(:, :, j)) * pi_BG(j);
            end
            hij = joint ./ sum(joint, 2);
            loglld(i) = sum(log(sum(joint,2)));
        
        %     M-step
        %     update
            pi_BG = sum(hij) / BG_samples;
            mu_BG = (hij' * TrainsampleDCT_BG) ./ sum(hij)';
            for j = 1:C
                sigma_BG(:,:,j) = diag(diag((TrainsampleDCT_BG - mu_BG(j, :))' ...
                    .* hij(:, j)' * (TrainsampleDCT_BG - mu_BG(j,:)) ./ sum(hij(:, j),1))+0.000001);
            end
        
            if i > 1
                if abs(loglld(i) - loglld(i-1)) < 0.001
                    break;
                end
            end
        end


        for i = 1:length(dim_list)
            dim = dim_list(i);
            A_mask = zeros(255*270, 1);
        
            for x = 1:length(A)
                p_BG = 0;
                p_FG = 0;
        
                for y = 1:C
                    p_BG = p_BG + mvnpdf(A(x,1:dim), mu_BG(y,1:dim), sigma_BG(1:dim,1:dim,y)) * pi_BG(y);
                end
        
                for y = 1:C
                    p_FG = p_FG + mvnpdf(A(x,1:dim), mu_FG(y,1:dim), sigma_FG(1:dim,1:dim,y)) * pi_FG(y);
                end
                
                if p_BG < p_FG
                    A_mask(x) = 1;
                end
            end
        
            resultImage = zeros(255, 270);
            for k=1:255
                resultImage(k, :) = A_mask((k-1)*270+1 : k*270)';
            end
        
            error = 0;
            for x = 1:255
                for y = 1:270
                    if mask(x,y) ~= resultImage(x,y)
                        error = error + 1;
                    end
                end
            end
            error_list(i) = error / (255 * 270);
            disp("idxFG " + a + " idxBG " + b + " dim " + dim);
        end
        hold on;
        plot(dim_list,error_list,'o-','linewidth',1,'markersize',5)
    end
    
    legend('BG1','BG2','BG3','BG4','BG5')
    fnm = sprintf('new_test_%d.fig',a);
    savefig(fnm)
    close;
end