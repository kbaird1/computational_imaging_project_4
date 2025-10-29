close all
clear all
clc

%% Load the images and the model

%load the STL10 dataset. Note: I have taken only the first 10,000 images
%from the dataset. I converted them to grayscale and resized them to 64x64
%pixels. Finally, I scaled them all to have pixel values between [0 1]
load('STL10_64.mat')

%Load the Hadamaard mask matrix. This is a 4096x4096 binary mask containing
%values of -1 or 1. In practice, each mask would require two acquisitions
%(since the DMD only can producec values of 0 and 1). This matrix was
%produced with the algorithm from: 
%
%Sun, M. J., Meng, L. T., Edgar, M. P., Padgett, M. J., & Radwell, N. 
%(2017). A Russian Dolls ordering of the Hadamard basis for compressive 
%single-pixel imaging. Scientific reports, 7(1), 3464.
load('RDHadamard_4096.mat')


%% Downsize the forward model

compressionRatio = 8; %How much data are we throwing out?
rowsToKeep = round(size(H,1))/compressionRatio; %These are the actual measurements we are keeping

H = H(1:rowsToKeep,:);

%% Produce the data
Y_vec = reshape(Y,size(Y,1),size(Y,2)*size(Y,3)); %Rreshape Y for matrix multiplication
X_vec = (H*Y_vec')'; %Apply the forward model to get our acquired data. The transposes are just because the ordering of the dimennsions of the matrices is a bit weird

% X_vec = H_vec + rand(size(H_vec))
%% Plot a random image and the acquired data
numImages = size(Y,1);

%Pick a random index to view
index = ceil(rand(1)*numImages);

%Plot the image
figure(1)
subplot(1,2,1)
imagesc(squeeze(Y(index,:,:)), [0 1])
colormap(gray)
axis image
xlabel('Y (pixels)')
ylabel('X (pixels)')
title(['Ground Truth Image ' num2str(index)]);

subplot(1,2,2)
plot(squeeze(X_vec(index,:)), 'k-', 'linewidth', 2)
xlabel('Acquisition Number')
ylabel('Signal (a.u.)')
axis square
title(['Acquired Data for Image ' num2str(index)]);

%% Get an initial guess of the image

X = (H'*X_vec')'/size(Y,2)/size(Y,3); %The inverse of a Hadamard matrix is its transpose (with some additional normalization)

X = reshape(X,size(Y,1),size(Y,2),size(Y,3));

%% Plot the initial guess of the image

figure(2)
subplot(1,3,1)
imagesc(squeeze(Y(index,:,:)), [0 1])
colormap(gray)
axis image
xlabel('Y (pixels)')
ylabel('X (pixels)')
title(['Ground Truth Image ' num2str(index)]);

subplot(1,3,2)
imagesc(squeeze(X(index,:,:)), [0 1])
colormap(gray)
axis image
xlabel('Y (pixels)')
ylabel('X (pixels)')
title(['Initial Guess Image ' num2str(index)]);

subplot(1,3,3)
imagesc(squeeze(abs(Y(index,:,:)-X(index,:,:))), [0 .5])
colormap(gray)
axis image
xlabel('Y (pixels)')
ylabel('X (pixels)')
title(['Absolute Error for Image ' num2str(index)]);

%% Save the data to load into Python
filename = 'Project4_Data.mat';

save(filename, 'X', 'Y', 'X_vec', 'H');


