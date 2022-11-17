%% Master Function
I = imread("peppers.png");
scale = 1/4;
[hpsf, cest, alphest] = blur_kernel_estimation(I, scale);
[deblurring_kernel] = deblur_kernel_estimation(hpsf);
significany = 3; 
[deblurred_image] = deblur(I, deblurring_kernel, alphest, significany);
subplot(1,2,1)
imshow(I)
title('Natural Blurred Image')
subplot(1,2,2)
imshow(deblurred_image)
title('Deblurred Image')

%% Blur Kernal Estimation
function [hpsf, cest, alphest] = blur_kernel_estimation(I, scale)
%  image downsampling using MaxPol-LPF
[rows, cols, ch] = size(I);
padbox = zeros(1, 2);
if mod(rows, 2) == 0
    padbox(1, 1) = 1;
end
if mod(cols, 2) == 0
    padbox(1, 2) = 1;
end
I_pad = padarray(I, padbox);
load('maxpol_lpf.mat')

max_skip = 5;
if 1/scale > max_skip
    N = log(1/scale) / log(max_skip);
    N = round(N);
    skip = (1/scale)^(1/N);
else
    skip = 1/scale;
    N = 1;
end
for iteration = 1: N
    [rowd, cold, chd] = size(I_pad);
    indi = round([1: skip: rowd]);
    indj = round([1: skip: cold]);
    %  lowpass filter
    I_pad = imfilter(I_pad, h_dowsampling(:));
    %  down-sampling
    I_pad = uint8(I_pad(indi, indj, :));
end
% calculate the image spectrum of images of original and degraded models
n_bins = 12;
[spect_scan_original, r_grid_bin] = radSpec(I_pad, n_bins);
[spect_scan_scaled, ~] = radSpec(I_pad, n_bins);
R = @(amp, alpha, c1, c2, x) amp*(exp(-alpha^2*x.^2/2) + c1*x)./(1/scale*exp(-alpha^2*x.^2/2/(1/scale)^2) + c2*x);
beta = 1.8;
fo = fitoptions('Method','NonlinearLeastSquares','Lower',[0, 0.75, 0, 0],'Upper',[Inf, 6, 4, 4],'StartPoint',[1 1 .05 .05]);
g = fittype(R,'coeff',{'amp', 'alpha', 'c1', 'c2'},'options',fo);
for channel = 1: ch
    a = spect_scan_original(:, channel);
    b = spect_scan_scaled(:, channel);
    h_estimate_fft(:, channel) = a./b;
    Omega = [r_grid_bin];
    FO = fit(Omega(:), h_estimate_fft(:, channel), g); % 'Trust-Region'
    ampest(channel) = FO.("amp");
    alphest(channel) = FO.("alpha");
    cest(channel) = FO.("c1");
    A = alphest(channel) * sqrt(gamma(1/beta) / gamma(3/beta));
    hpsf(:,channel) = ampest(channel)/(2*gamma(1+1/beta)*A)*exp(-abs(([-1,0,1])/A).^beta);
end
end
%%  Deblurring Kernal Estiation
function [deblurring_kernel] = deblur_kernel_estimation(hpsf)
% Gaussian Kernal
maxamp = 30; 
N = 16;
l_deblurring = 32;
pctuoff = 2*l_deblurring-0;
% l = (size(h_psf,1)-1)/2;
for channel = 1: size(hpsf, 2)
    blurring_kernel = hpsf(:, channel)/sum(hpsf(:, channel));
    [alpha_estimated] =  spectrum_fit(blurring_kernel, N,maxamp);
    clear d 
    d(1, :) = derivcent(l_deblurring, 2*l_deblurring, 0, 0, false);
    for n = 1: floor(N/2)
        d(n+1, :) = alpha_estimated(n+1)/alpha_estimated(1)*derivcent(l_deblurring, pctuoff, 0, 2*n, false);
    end
    deblurring_kernel(:, channel) = sum(d(2: end, :), 1);
end
end
%% One Shot poll
function [deblurred_image] = deblur(Img, deblurring_kernel, alpha_estimate, significany)
[rows, cols, ch] = size(Img);
for channel = 1: ch
    beta = 1.8;
    sigm = alpha_estimate(channel)*0.85;
    A = sigm * sqrt(gamma(1/beta) / gamma(3/beta));
    h_GGaussian = 1/(2*gamma(1+1/beta)*A)*exp(-abs(([-1,0,1])/A).^beta);
    % generate inverse deconvolution filter
    inverse_deblurring_kernel = conv(deblurring_kernel(:, channel), h_GGaussian(:));    
    %  apply inverse deblurring kernel on image observation
    deblurring_edges = imfilter(Img(:,:,channel), inverse_deblurring_kernel);
    gamma_significance(channel) = entropy(Img(:,:,channel))/(entropy(deblurring_edges)+0.5)*significany;
    deblurred_image(:,:,channel) = Img(:,:,channel) + gamma_significance(channel) * deblurring_edges;
end
%  convert double format into uint8 format
deblurred_image = im2uint8(deblurred_image);
end
%% Spectrum
function [spec, r_grid_bin] = radSpec(inputImage, n_bins)
[rows, cols, ch] = size(inputImage);
if mod(rows, 2) == 0
    inputImage = padarray(inputImage, [1, 0]);
end
r_max = floor(max([rows, cols])/2);
for channel = 1: ch
    I = inputImage(:,:,channel);
    FT = fftshift(fft2(I, 2*r_max+1, 2*r_max+1));
    frequency_spectrum = abs(FT);
    h_bin = 1/n_bins;
    r_grid_bin = [h_bin:h_bin:1]'*pi;
    r_grid = [-r_max: r_max]/r_max*pi;
    [X, Y] = meshgrid(r_grid);
    Y = -Y;
    A = sqrt(X.^2 + Y.^2);    
    circle_mask = A <= pi;
    pixel_area = pi^3/sum(circle_mask(:));
    for iteration = 1: n_bins
        mask_out =  A <= r_grid_bin(iteration);
        if iteration ~= 1
            mask_in = A <= r_grid_bin(iteration-1);
            mask = xor(mask_out, mask_in);
        else
            mask = mask_out;
        end
        S(iteration) = pixel_area * sum(mask(:));
        spec(iteration, channel) = sum(frequency_spectrum(mask))/S(iteration);
    end
    spec(:, channel) = spec(:, channel)/spec(1, channel);
end
end
%% Spectrum Fit
function [alphestd] = spectrum_fit(blurringkernel, N, maxamp)
omg = [-pi: 2*pi/(1024): pi];
kerspec = fftshift(fft(blurringkernel, 1024));
yful = 1./kerspec;
%  polynomial fitting on the frequency domain
indx = abs(yful) < maxamp;
omgslt = omg(indx);
yslt = yful(indx);
yslt = abs(yslt);
for n = 0: floor(N/2)
    polyTerms{n+1} = ['(-1)^', num2str(n), '*x^',num2str(2*n)];
    polyCoeffs{n+1} = ['a',num2str(2*n)];
end
ft = fittype(polyTerms, 'coefficients', polyCoeffs);
w = ones(size(omgslt));
FO = fit(omgslt(:), yslt(:), ft, 'Weights', w); % 'Trust-Region'
for n = 0: floor(N/2)
    alphestd(n+1) = FO.("a"+num2str(2*n));
end
end