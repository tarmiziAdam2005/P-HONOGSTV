clc;
clear all;
close all;

NN = 10;
sumPsnr = 0;
sumSSIM =0;
sumSnr  = 0;
sumIt   = 0; % to calculate average itration to converge
sumTime = 0;

saveRes = 'no';


imageName = 'foot.png';


Img = imread(imageName); %Your Image goes here

if size(Img,3) > 1
    Img = rgb2gray(Img);
end

N = numel(Img);

[row, col] = size(Img);

row = int2str(row);
col = int2str(col);

imageSize = [row 'x' col];

%*******************************Kernels for deblurring********************
%K = fspecial('gaussian', [7 7], 5); % Gaussian Blur
                                     
                                     
%K = fspecial('motion', 30, 25); % Motion blur




%************ For poisson noise removal ***********

Img  = double(Img);
max_value = max(max(Img));
img_max  = 255;   % or 400

Img  = Img/max_value*img_max;



K     =   fspecial('average',9); % For denoising
%K     = fspecial('gaussian',7,5);
f = imfilter(Img,K,'circular');


f = poissrnd(f);   %add poisson noise

% *************************************************

%**************Uncomment for denoising*********************
opts.lam       = 430; % for denoising try starting with these 
opts.omega     = 20;  
opts.grpSz     = 3; % OGS group size
opts.Nit       = 100;
opts.Nit_inner = 5;
opts.tol       = 1e-3;
opts.p         = 0.1;
opts.maxPix    = img_max;


out = Prop_Poiss(f, Img, K,opts);  %% Main solver

%Some result options

figure;
imshow(f,[]);
title(sprintf('Noisy(PSNR = %3.3f dB,SSIM = %3.3f, SNR = %3.3f) ',...
                       psnr_fun(f,Img),ssim_index(f,Img),snr_fun(f, Img)));

figure;
imshow(out.sol,[])
title(sprintf('ADMM-TV Deblurred (PSNR = %3.3f dB,SSIM = %3.3f, SNR = %3.3f ) ',...
                       psnr_fun(out.sol,Img),ssim_index(out.sol, Img),snr_fun(out.sol, Img) ));
                   
                   