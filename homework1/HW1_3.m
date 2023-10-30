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

A = zeros(255, 270);

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
        
        % pick position of 2nd largest magnitude as the feature value;
        [secondLargeNum, secondLargeIdx] = maxk(flatBlock, 2);

        % use BDR to find class Y for each block;
        % create a binary mask;
        if priorFG * freqFG(secondLargeIdx(2)) > priorBG * freqBG(secondLargeIdx(2))
            A(row, col) = 1;
        else
            A(row, col) = 0;
        end
    end
end

imshow(uint8(A), [0 1]);
savefig('A.fig');