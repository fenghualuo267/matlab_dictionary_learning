function [ P ] = extract_patches( I, Np, win )  %提取图像块
    % get image dimensions（尺寸）
    width  = size( I, 2 );
    height = size( I, 1 );

    % window offset from central pixel, patch will be win pixels either side of
    % the central pixel. pwidth is the width of the patch, in pixels and N
    % is the number of pixels in the patch
    %窗口从中心像素开始偏移，patch将获得以中心像素为中心的矩形窗。pwidth是patch的宽度，单位是像素，N是patch的像素个数
    pwidth = 2*win+1;
    N      = pwidth^2;

    % patch center location（patch中心像素）, here assuming an MxM array of output patches（这里假设一个MxM数组的输出patch）, 
    % we generate（产生） a Np=M*M patch center location（位置） randomly, 
    % constraining（约束） the centers to be at least（至少） win pixels from the image border（边界） to avoid edge cases（避免边缘情况）
    % 
    px = randi( [win+1,width-win-1],  Np, 1 );
    py = randi( [win+1,height-win-1], Np, 1 );

    % Y will store（保存） the patch dictionary, with each（每个） patch packed（包装） as a column（列） in
    % the N*Np matrix（矩阵）. Each patch pixel is looped（成圈的） over and the appropriate（合适的） row（行）
    % of Y is generated（产生） via（通过） the interp2 function using |nearest neighbor interpolation（近邻插值法）
    % 
    P = zeros( N, Np );
    id = 1;
    for i=-win:win,
        for j=-win:win,
            P( id, : ) = interp2( I, px+i, py+j, 'nearest' );
            id = id+1;
        end
    end
    
    % ugly hack here...
    %P(isnan(P)) = 0.0;
    %P(isinf(P)) = 0.0;
end
