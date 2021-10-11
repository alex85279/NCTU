//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
// Date: Nov 2020
//
#ifndef __SOUND_MANAGER__
#define __SOUND_MANAGER__

#include "sound.h"

#define SOUND_MNG_DFT_NUM_SOUNDS    4

class SOUND_MANAGER {
private:
    static SOUND_MANAGER *_mInstance;
protected:
     int mNumSounds;
     //
     SOUND **mSound_Explosion;
     int mSoundIndex_Explosion;
     //
     SOUND **mSound_Fire;
     int mSoundIndex_Fire;
     //
     SOUND **mSound_Stamina;
     int mSoundIndex_Stamina;
     //
     SOUND **mSound_LevelUp;
     int mSoundIndex_LevelUp;
     
     //
     void init(int numSounds = SOUND_MNG_DFT_NUM_SOUNDS);
public:
    SOUND_MANAGER();
    static SOUND_MANAGER *getInstance() {
        if (!_mInstance) {
            _mInstance = new SOUND_MANAGER;
            //_mInstance->init();
        }
        return _mInstance;
    }
    void play_Explosion();
    void play_Fire();
    void play_Stamina();
    void play_LevelUp();
};

#endif