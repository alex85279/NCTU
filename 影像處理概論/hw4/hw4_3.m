RGB = imread('HW4_einstein.tif'); %ÅªÀÉ
fft_RGB = fft2(RGB);
fft_RGB = fftshift(fft_RGB);
[M N]=size(fft_RGB);

P_1=10;  
P_2=20; 
P_3=30; 
newX=0:N-1;
newY=0:M-1;
[newX newY]=meshgrid(newX,newY);
fix_X=0.5*N;
fix_Y=0.5*M;
filter1=exp(-((newX - fix_X).^2 + (newY - fix_Y).^2)./(2 * P_1).^2);
filter2=exp(-((newX - fix_X).^2 + (newY - fix_Y).^2)./(2 * P_2).^2);
filter3=exp(-((newX - fix_X).^2 + (newY - fix_Y).^2)./(2 * P_3).^2);

output1 = fft_RGB.*filter1;
output1 = ifftshift(output1); 
output1 = ifft2(output1); 

output2 = fft_RGB.*filter2;
output2 = ifft2(ifftshift(output2));

output3 = fft_RGB.*filter3;
output3 = ifft2(ifftshift(output3));

imshow(output1,[])
title('Filter Parameter=10')

figure, imshow(output2,[])
title('Filter Parameter=20')

figure, imshow(output3,[])
title('Filter Parameter=30')

[M, N] = size(RGB);
zero_padding = padarray(RGB, [M, N], 0, 'post');
fft_RGB=fftshift(fft2(zero_padding));
[M,N]=size(fft_RGB);
newX=0:N-1;
newY=0:M-1;
[newX,newY]=meshgrid(newX,newY);
Cx=0.5*N;
Cy=0.5*M;
filter4=exp(-((newX-Cx).^2+(newY-Cy).^2)./(2*P_3).^2);
output4=fft_RGB.*filter4;
output4=ifft2(ifftshift(output4));
[M, N] = size(RGB);
figure, imshow(output4(1:M,1:N),[])
title('with padding, filter parameter=15')