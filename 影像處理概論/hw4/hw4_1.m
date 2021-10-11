RGB = imread('HW4_einstein.tif');
fft_RGB = fft2(RGB);

% get magnitude of DFT Image
fft_Mag = abs(fft_RGB);
fft_Mag= log(fft_Mag+1);
fft_Mag = mat2gray(fft_Mag);

% get phase of DFT Image
fft_phase = angle(fft_RGB); 
fft_phase = mat2gray(fft_phase); 

% display magnitude and phase
imshow(fft_Mag,[]); %display
title('magnitude')
figure, imshow(fft_phase,[]); 
title('phase')