clc
clear
addpath('./GCMex')

img1= double(imread('left.png'));
img2 = double(imread('right.png'));

disp = stereoGC(img1, img2, 50, 4);
imshow(disp)

function result = stereoGC(img_l, img_r, D_MAX, adjacency_num)
    [H, W, C] = size(img_l);
    d = 0 : D_MAX;
    N = H * W;
    unary = ones(D_MAX+1, N);
    [I, J] = meshgrid(1:D_MAX+1, 1:D_MAX+1);
    labelcost = min(25, (I - J).*(I - J));
    labelcost = labelcost./ 25;

    for i = 1 : D_MAX+1
        I = zeros(H,W,C);
        I(:, 1:W-d(i),:) = img_l(:, d(i)+1:W, :);
        D = sqrt(sum((img_r-I).^2, 3));
        D = reshape(D, 1, N);
        unary(i,:) = D;
    end
    unary = unary./ max(max(unary)) * 4;
    if adjacency_num == 8
        unary = unary./ max(max(unary)) * 8;
    end
    [~,segclass] = min(unary,[],1);

    loctmp = ones(H, W);
    top = find(imtranslate(loctmp, [0, 1], 'FillValues', 0) ~= 0);
    bottom = find(imtranslate(loctmp, [0, -1], 'FillValues', 0) ~= 0);
    left = find(imtranslate(loctmp, [1, 0], 'FillValues', 0) ~= 0);
    right = find(imtranslate(loctmp, [-1, 0], 'FillValues', 0) ~= 0);
    if adjacency_num == 8
        lefttop = find(imtranslate(loctmp, [1, 1], 'FillValues', 0) ~= 0);
        righttop = find(imtranslate(loctmp, [-1, 1], 'FillValues', 0) ~= 0);
        leftbottom = find(imtranslate(loctmp, [1, -1], 'FillValues', 0) ~=0);
        rightbottom = find(imtranslate(loctmp, [-1, -1], 'FillValues', 0) ~= 0);
    end
    m = [right;left;top;bottom];
    n = [left;right;bottom;top];
    if adjacency_num == 8
        m = [right;left;top;bottom;righttop;lefttop;rightbottom;leftbottom];
        n = [left;right;bottom;top;leftbottom;rightbottom;lefttop;righttop];
    end
    pairwise = sparse(m, n, 1);
    [labels, ~, ~] = GCMex(segclass-1, single(unary), pairwise, single(labelcost), 1);
    dp = labels;
    dp = dp / max(dp);
    result = reshape(dp, H, W);
end
