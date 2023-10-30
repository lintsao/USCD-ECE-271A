mask = imread('cheetah_mask.bmp');
errorCount = 0;
sizes = size(mask);
rows = sizes(1);
cols = sizes(2);

% check whether the predicted label equals to the ground truth label;
for row  = 1 : rows
    for col  = 1 : cols
        if (mask(row, col) / 255 ~= A(row, col))
            errorCount = errorCount + 1;
        end
    end
end


% compute error rate;
errorRate = errorCount / (rows * cols);
disp(errorRate);