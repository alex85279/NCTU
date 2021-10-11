
#include "ppm2bmp.h"
#include<iostream>
typedef unsigned char BYTE;
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef long LONG;

typedef struct tagBITMAPINFOHEADER {
	DWORD biSize;
	LONG  biWidth;
	LONG  biHeight;
	WORD  biPlanes;
	WORD  biBitCount;
	DWORD biCompression;
	DWORD biSizeImage;
	LONG  biXPelsPerMeter;
	LONG  biYPelsPerMeter;
	DWORD biClrUsed;
	DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;

typedef struct tagRGBQUAD {
	BYTE rgbBlue;
	BYTE rgbGreen;
	BYTE rgbRed;
	BYTE rgbReserved;
} RGBQUAD;

typedef struct tagBITMAPINFO {
	BITMAPINFOHEADER bmiHeader;
	RGBQUAD          bmiColors[1];
} BITMAPINFO, *PBITMAPINFO;

typedef struct tagBITMAPFILEHEADER {
	WORD  bfType;
	DWORD bfSize;
	WORD  bfReserved1;
	WORD  bfReserved2;
	DWORD bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

ErrorID ppm2bmp(char *ppmFilename, char *bmpFilename) {
	ErrorID err = errOK;
	FILE *fpPpm = NULL;
	FILE *fpBmp = NULL;
	BYTE *bufPpm = NULL;
	BYTE *bufBmp = NULL;

	// check params
	if (!errParamError || !bmpFilename) {
		err = errParamError;
		goto FINISH;
	}

	printf("ppm file = %s, bmp file = %s\n", ppmFilename, bmpFilename);

	//--- read ppm file ---

	// open ppm file 
	fpPpm = fopen(ppmFilename, "rb");
	if (!fpPpm){
		printf("read ppm file %s failed.\n", ppmFilename);
		err = errOpenPpmError;
		goto FINISH;
	}

	// read ppm file header
	int w, h, c;
	(void)fscanf(fpPpm, "P3\n%d %d\n%d\n", &w, &h, &c);
	printf("ppm file %s, w=%d, h=%d, c=%d\n", ppmFilename, w, h, c);

	// only deal with 24bit ppm (each pixel cost 3 bytes)
	if (c!=0xFF){
		printf("can not handle ppm file %s.\n", ppmFilename);
		err = errOpenPpmError;
		goto FINISH;
	}

	// read ppm file buffer
	int bufPpmSize = w * h * 4;
	bufPpm = (BYTE *)malloc(bufPpmSize);
	if (!bufPpm){
		err = errMemoryError;
		goto FINISH;
	}
	memset(bufPpm, 0, bufPpmSize);
	if (bufPpmSize != fread(bufPpm, 1, bufPpmSize, fpPpm)){
		err = errOpenBmpError;
		goto FINISH;
	};

	//--- convert ppm buffer to bmp buffer ---

	int widthStep = (((w * 24) + 31) & (~31)) / 8; // line width of buffer % 4 == 0
	
	int bufBmpSize = widthStep * h;
	bufBmp = (BYTE *)malloc(bufBmpSize);
	if (!bufBmp){
		err = errMemoryError;
		goto FINISH;
	}
	memset(bufBmp, 0, bufBmpSize);
	
	// seems ppm data is bottom to top
	BYTE *pp = bufPpm;
	BYTE *pb = bufBmp + bufBmpSize;
	int hh = h;
	while (hh > 0){
		pb -= widthStep;
		memcpy(pb, pp, w * 3);
		pp += w * 3;

		hh--;
	}

	//--- write bmp file ---

	// bmp info
	BITMAPINFO bitmapinfo;
	memset(&bitmapinfo, 0, sizeof(BITMAPINFO));
	bitmapinfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapinfo.bmiHeader.biWidth = w;
	bitmapinfo.bmiHeader.biHeight = h;
	bitmapinfo.bmiHeader.biPlanes = 1;
	bitmapinfo.bmiHeader.biBitCount = 24;
	bitmapinfo.bmiHeader.biXPelsPerMeter = 0;
	bitmapinfo.bmiHeader.biYPelsPerMeter = 0;
	bitmapinfo.bmiHeader.biSizeImage = bufBmpSize;
	bitmapinfo.bmiHeader.biClrUsed = 0;
	bitmapinfo.bmiHeader.biClrImportant = 0;

	// bmp file header
	BITMAPFILEHEADER bmpHeader;
	memset(&bmpHeader, 0, sizeof(BITMAPFILEHEADER));
	bmpHeader.bfType = 0x4D42;
	bmpHeader.bfOffBits = sizeof(BITMAPINFOHEADER) + sizeof(BITMAPFILEHEADER);
	bmpHeader.bfSize = bmpHeader.bfOffBits + bufBmpSize;

	// open bmp file
	fpBmp = fopen(bmpFilename, "wb");
	if (!fpBmp){
		printf("bmp file %s open failed.\n", bmpFilename);
		err = errWriteBmpError;
		goto FINISH;
	}

	// write bmp file
	fwrite(&bmpHeader, 1, sizeof(BITMAPFILEHEADER), fpBmp);
	fwrite(&(bitmapinfo.bmiHeader), 1, sizeof(BITMAPINFOHEADER), fpBmp);
	fwrite(bufBmp, 1, bufBmpSize, fpBmp);

FINISH:
	if (bufPpm) free(bufPpm);
	if (bufBmp) free(bufBmp);
	if (fpPpm) fclose(fpPpm);
	if (fpBmp) fclose(fpBmp);

	return err;
}

int main(int argc, char *argv[]) {
	// make sure args count ok
	ppm2bmp("../ray.ppm", "../ray.bmp");

	return 0;
}