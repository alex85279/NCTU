/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/

#ifndef __DATA_READER_H__
#define __DATA_READER_H__

#include <string>

class MESH_INFO {
public:
    MESH_INFO( );
    std::string mMeshName;
    double mMeshScale;
    double mDegreeCorrection;
};

class DATA_READER {
protected:
	static bool mEnableExpFog;
	static bool mEnableShadow;
	static float mExpFogDensity;
	static int mMaxBulletsNum;
	static int mMaxMonstersNum;
    static float mBulletSpeed;
	static double mWaterCoord_Y;
	static std::string mWaterMaterialName;
	//static std::string mMeshName;
    //static double mMeshScale;
    static MESH_INFO mMeshInfo;
    //
    static std::string mAvatarMesh;
    static double mAvatarEyePosition_Y;
    static double mAvatar_WalkingMaxSpeed;
    //
    static std::string mSoundFile_Explosion;
    static std::string mSoundFile_Fire;
    static std::string mSoundFile_Stamina;
    static std::string mSoundFile_LevelUp;
public:
	DATA_READER();
	static void readData();
	static bool isEnabledShadow();
	static bool isEnabledExpFog();
	static float getExpFogDensity();
	static float getBulletSpeed();
	static int getMaxBulletsNum();
	static int getMaxMonstersNum();
	static double getWaterCoord_Y();
	static std::string getWaterMaterialName();
static std::string getMeshName();
static double getMeshScale();
//
static double getAvatarEyePosition_Y();
static std::string getAvatarMeshName();
static double getAvatarWalkingMaxSpeed();
//
static std::string getSoundFileName_Explosion();
static std::string getSoundFileName_Fire();
static std::string getSoundFileName_Stamina();
static std::string getSoundFileName_LevelUp();

//
static void report();
};

#endif