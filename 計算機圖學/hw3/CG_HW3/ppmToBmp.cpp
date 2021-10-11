#include<math.h>
#include <iomanip> 
#include <stdlib.h>
#include <windows.h>
#include <stdio.h>

typedef struct BGR
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
}BGR;

typedef struct RGB
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
}RGB;

void PPMtoBMP(char *pFramePPM,int bpp)
{
    FILE* pPPM;
    fopen_s(&pPPM, pFramePPM, "rb");

    int width, height;
    char header[20];
    fgets(header, 20, pPPM);// get "P6" 
    fgets(header, 20, pPPM);// get "width height" 
    sscanf_s(header, "%d %d\n", &width, &height);
    fgets(header, 20, pPPM);// get "255" 

    FILE *fp;
    fopen_s(&fp, "ray.bmp", "wb");
    if (fp == NULL)
    {
        printf("file is null.\n");
        return;
    }

    BITMAPFILEHEADER bmpheader;
    BITMAPINFOHEADER bmpinfo;
    RGB *ppmBitsRGB = (RGB *)malloc(width*height * sizeof(RGB));

    bmpheader.bfType = 0x4d42;
    bmpheader.bfReserved1 = 0;
    bmpheader.bfReserved2 = 0;
    bmpheader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) - 2;
    bmpheader.bfSize = bmpheader.bfOffBits + width * height*bpp / 8;

    bmpinfo.biSize = sizeof(BITMAPINFOHEADER);
    bmpinfo.biWidth = width;
    bmpinfo.biHeight = height;
    bmpinfo.biPlanes = 1;
    bmpinfo.biBitCount = bpp;
    bmpinfo.biCompression = 0;//BI_RGB;
    bmpinfo.biSizeImage = width* height*bpp / 8;
    bmpinfo.biXPelsPerMeter = 100;
    bmpinfo.biYPelsPerMeter = 100;
    bmpinfo.biClrUsed = 0;
    bmpinfo.biClrImportant = 0;

    fwrite(&bmpheader.bfType, sizeof(bmpheader.bfType), 1, fp);
    fwrite(&bmpheader.bfSize, sizeof(bmpheader.bfSize), 1, fp);
    fwrite(&bmpheader.bfReserved1, sizeof(bmpheader.bfReserved1), 1, fp);
    fwrite(&bmpheader.bfReserved2, sizeof(bmpheader.bfReserved2), 1, fp);
    fwrite(&bmpheader.bfOffBits, sizeof(bmpheader.bfOffBits), 1, fp);

    fwrite(&bmpinfo.biSize, sizeof(bmpinfo.biSize), 1, fp);
    fwrite(&bmpinfo.biWidth, sizeof(bmpinfo.biWidth), 1, fp);
    fwrite(&bmpinfo.biHeight, sizeof(bmpinfo.biHeight), 1, fp);
    fwrite(&bmpinfo.biPlanes, sizeof(bmpinfo.biPlanes), 1, fp);
    fwrite(&bmpinfo.biBitCount, sizeof(bmpinfo.biBitCount), 1, fp);
    fwrite(&bmpinfo.biCompression, sizeof(bmpinfo.biCompression), 1, fp);
    fwrite(&bmpinfo.biSizeImage, sizeof(bmpinfo.biSizeImage), 1, fp);
    fwrite(&bmpinfo.biXPelsPerMeter, sizeof(bmpinfo.biXPelsPerMeter), 1, fp);
    fwrite(&bmpinfo.biYPelsPerMeter, sizeof(bmpinfo.biYPelsPerMeter), 1, fp);
    fwrite(&bmpinfo.biClrUsed, sizeof(bmpinfo.biClrUsed), 1, fp);
    fwrite(&bmpinfo.biClrImportant, sizeof(bmpinfo.biClrImportant), 1, fp);

    int y,x;
    for (y = 0; y < height; y++)
    {
        fread(&ppmBitsRGB[y*width], 3 * width, 1, pPPM);
        for (x = width - 1; x >= 0; x--)
        {
            fwrite(&ppmBitsRGB[y*width + x].b, 1, 1, fp);
            fwrite(&ppmBitsRGB[y*width + x].g, 1, 1, fp);
            fwrite(&ppmBitsRGB[y*width + x].r, 1, 1, fp);
        }
    }
    fclose(fp);
    fp = NULL;
    return;
}

void BMPtoPPM(char *pFrameRGB)
{
    FILE* pBMP;
    fopen_s(&pBMP, pFrameRGB, "rb");
    BITMAPINFOHEADER infoHeader;
    BITMAPFILEHEADER fileHeader;
    FILE *pFile;
    fopen_s(&pFile, "test.ppm", "wb");
    if (pFile == NULL)
    {
        printf("file is null.\n");
        return;
    }
    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pBMP);
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pBMP);

    if (infoHeader.biBitCount != 24){
        printf("it is not a 24-bit rgb image.\n");
        return;
    }
    long width = infoHeader.biWidth;
    long height = infoHeader.biHeight;

    // Write header
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    int i, j;
    BGR *bmpBitsBGR = (BGR *)malloc(width*height * sizeof(BGR));

    //?取每一?像素?的BGR值
    fseek(pBMP, fileHeader.bfOffBits, 0);
    for (i = 0; i <height; i++)
    {

        fread(&bmpBitsBGR[i*width], 3 * width, 1, pBMP);
        for (j = width - 1; j >= 0; j--)
        {
            fwrite(&bmpBitsBGR[i*width + j].r, 1, 1, pFile);
            fwrite(&bmpBitsBGR[i*width + j].g, 1, 1, pFile);
            fwrite(&bmpBitsBGR[i*width + j].b, 1, 1, pFile);
        }
        fseek(pBMP, (4 - (width % 4)) % 4, SEEK_CUR); //4字???
    }

    // Close file
    fclose(pBMP);
    fclose(pFile);
    free(bmpBitsBGR);
    bmpBitsBGR = NULL;
    return;
}

void main(){
    char readPath[] = "ray.ppm";
    PPMtoBMP(readPath,24); 
    return;
}
