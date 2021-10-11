/*
This is a game demo written by Sai-Keung Wong.
Date: Dec 2006 - May 2011
Email: wingo.wong@gmail.com
*/
#ifndef __SOUND_H__
#define __SOUND_H__
#include <string>
#include <windows.h>
#include "Framework.h"
#include "CWaves.h"

#define NUMBUFFERS              (4)

//control sound
class SOUND {
private:

	
protected:
		ALuint		    uiBuffers[NUMBUFFERS];
	ALuint		    uiSource;
	ALuint			uiBuffer;
	ALint			iState;
	CWaves *		pWaveLoader;
	WAVEID			WaveID;
	ALint			iLoop;
	ALint			iBuffersProcessed, iTotalBuffersProcessed, iQueuedBuffers;
	WAVEFORMATEX	wfex;
	unsigned long	ulDataSize;
	unsigned long	ulFrequency;
	unsigned long	ulFormat;
	unsigned long	ulBufferSize;
	unsigned long	ulBytesWritten;
	void *			pData;

	char *mFileName;

protected:
    bool loadSound(char *a_FileName);
public:
	SOUND();
	~SOUND();
	
    bool init(const std::string &soundFileName,
        bool flg_initOpenAL = true
        );
	bool init();
	bool play();
	bool isStopped() const;
};
#endif