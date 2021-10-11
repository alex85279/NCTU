/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/
#include <fstream>
#include <iostream>
#include <string>
#include "read_data.h"
#define DATA_FILE_NAME	"309553012_game_data.txt"

using namespace std;

MESH_INFO::MESH_INFO( ):
    mMeshName("penguin.mesh")
    , mMeshScale (1.0)
    , mDegreeCorrection(00.0)
{ 

}

bool DATA_READER::mEnableShadow = true;
int	DATA_READER::mMaxBulletsNum = 10;
int	DATA_READER::mMaxMonstersNum = 10;
float DATA_READER::mBulletSpeed = 200;
double	DATA_READER::mWaterCoord_Y = 0.0;
std::string DATA_READER::mWaterMaterialName = "";
//std::string DATA_READER::mMeshName = "";
//double DATA_READER::mMeshScale = 1.0;
bool DATA_READER::mEnableExpFog = true;
float DATA_READER::mExpFogDensity = 0.0003;
//
MESH_INFO DATA_READER::mMeshInfo;
//
std::string DATA_READER::mAvatarMesh = "robot.mesh";
     double DATA_READER::mAvatarEyePosition_Y = 120;
     double DATA_READER::mAvatar_WalkingMaxSpeed = 5;
//
     std::string DATA_READER::mSoundFile_Explosion = "";
     std::string DATA_READER::mSoundFile_Fire = "";
     std::string DATA_READER::mSoundFile_Stamina = "";
     std::string DATA_READER::mSoundFile_LevelUp = "";
//
DATA_READER::DATA_READER()
{

}

void DATA_READER::readData()
{
    cout << "Begin DATA_READER::readData()" << endl;
    using namespace std;
    std::ifstream *fp;
    fp = new std::ifstream(DATA_FILE_NAME, ios::in | ios::binary);
    if (fp == 0 || fp->fail()) {
        cout << "Cannot open data file:" << DATA_FILE_NAME << endl;
        return;
    }

    std::string key;
    double num;
    while (!fp->eof()) {
        *fp >> key;

        cout << "Reading:" << key <<endl;

        if (key.compare("NUM_BULLETS") == 0) {
            *fp >> num;
            mMaxBulletsNum = num;
        }

        if (key.compare("NUM_MONSTERS") == 0) {
            *fp >> num;
            mMaxMonstersNum = num;
        }

        if (key.compare("WATER_Y_COORD") == 0) {
            *fp >> num;
            mWaterCoord_Y = num;
        }

        if (key.compare("MESH_SCALE") == 0) {
            double meshScale;
            *fp >> meshScale;
            mMeshInfo.mMeshScale = meshScale;

        }

        if (key.compare("WATER_MATERIAL") == 0) {
            *fp >> mWaterMaterialName;

        }
        if (key.compare("MESH_NAME") == 0) {
            string meshName;
            *fp >> meshName;
            mMeshInfo.mMeshName = meshName;

            double degreeCorrection;
            *fp >> degreeCorrection;
            mMeshInfo.mDegreeCorrection = degreeCorrection;

        }

        if (key.compare("EXP_FOG_ENABLED") == 0) {
            *fp >> mEnableExpFog;
            *fp >> mExpFogDensity;
        }

        if (key.compare("SHADOW_ENABLED") == 0) {
            *fp >> mEnableShadow;

        }

        if (key.compare("AVATAR_EYE_POSITION_Y") == 0) {
            *fp >> mAvatarEyePosition_Y;

        }

        if (key.compare("AVATAR_MESH") == 0) {
            *fp >> mAvatarMesh;

        }

        if (key.compare("AVATAR_WALKING_MAX_SPEED") == 0) {
            *fp >> mAvatar_WalkingMaxSpeed;

        }

         if (key.compare("SOUND_WAVE_FILE_EXPLOSION") == 0) {
            *fp >> mSoundFile_Explosion;

        }

          if (key.compare("SOUND_WAVE_FILE_FIRE") == 0) {
            *fp >> mSoundFile_Fire;

        }

           if (key.compare("SOUND_WAVE_FILE_STAMINA") == 0) {
            *fp >> mSoundFile_Stamina;

        }

           if (key.compare("SOUND_WAVE_FILE_LEVEL_UP") == 0) {
            *fp >> mSoundFile_LevelUp;

        }
           

        key.clear(); // a must here
    }

    report();

    cout << "End DATA_READER::readData()" << endl;

}

bool DATA_READER::isEnabledShadow()
{
    return mEnableShadow;
}

int DATA_READER::getMaxBulletsNum()
{
    return mMaxBulletsNum;
    
}

float DATA_READER::getBulletSpeed( )
{
   return mBulletSpeed;
}

bool DATA_READER::isEnabledExpFog()
{
    return mEnableExpFog;
}

float DATA_READER::getExpFogDensity()
{
    return mExpFogDensity;
}


int DATA_READER::getMaxMonstersNum()
{
    return mMaxMonstersNum;
}

double DATA_READER::getMeshScale()
{
    return mMeshInfo.mMeshScale;
}
double DATA_READER::getWaterCoord_Y()
{
    return mWaterCoord_Y;
}

std::string DATA_READER::getWaterMaterialName()
{
    return mWaterMaterialName;
}

std::string DATA_READER::getMeshName()
{
    return mMeshInfo.mMeshName;
}

double DATA_READER::getAvatarEyePosition_Y(){
    return mAvatarEyePosition_Y;
}

std::string DATA_READER::getAvatarMeshName(){
    return mAvatarMesh;
}

double DATA_READER::getAvatarWalkingMaxSpeed(){
    return mAvatar_WalkingMaxSpeed;
}

std::string DATA_READER::getSoundFileName_Explosion(){
    return "explosion.wav";
}

 std::string DATA_READER::getSoundFileName_Fire() {
    return "Fireball.wav";
}

 std::string DATA_READER::getSoundFileName_Stamina() {
    return "warning_not_enough_stamina.wav";
}

  std::string DATA_READER::getSoundFileName_LevelUp() {
    return "level_up.wav";
}
 

void DATA_READER::report()
{
    cout << "DATA_READER::report" << endl;
    cout << "mEnableExpFog:" << "\t" << mEnableExpFog << endl;
    cout << "mEnableShadow:" << "\t" << mEnableShadow << endl;
    cout << "mExpFogDensity:" << "\t" << mExpFogDensity << endl;
    cout << "mMaxBulletsNum:" << "\t" << mMaxBulletsNum << endl;
    cout << "mMaxMonstersNum:" << "\t" << mMaxMonstersNum << endl;
    cout << "mBulletSpeed:" << "\t" << mBulletSpeed << endl;
    cout << "mWaterCoord_Y:" << "\t" << mWaterCoord_Y << endl;
    cout << "mExpFogDensity:" << "\t" << mExpFogDensity << endl;
    cout << "mWaterMaterialName:" << "\t" << mWaterMaterialName << endl;
    cout << "mMeshName:" << "\t" << mMeshInfo.mMeshName << endl;
    cout << "mMeshScale:" << "\t" << mMeshInfo.mMeshScale << endl;
    cout << "mDegreeCorrection:" << "\t" << mMeshInfo.mDegreeCorrection << endl;
    cout << "mAvatarMesh:" << "\t" << mAvatarMesh << endl;
    cout << "mAvatarEyePosition_Y:" << "\t" << mAvatarEyePosition_Y << endl;
    cout << "mAvatar_WalkingMaxSpeed:" << "\t" << mAvatar_WalkingMaxSpeed << endl;
}




