//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
// Date: Nov 2020
//
/*!
\brief 3D Game Programming
\n
My Name: AA BB CC
\n
My ID: 0123456789
\n
My Email: aaa@cs.nctu.edu.tw

This is an assignment of 3D Game Programming

*/

#ifndef __BasicTutorial_00_h_
#define __BasicTutorial_00_h_

#include "BaseApplication.h"
#include "main_char.h"
#include "monster_manager.h"
#include "sound.h"
#include "bar2D.h"
#include "digit_string_dialogue.h"
#include "wago_ogre_mesh.h"
#include "WeaponParticleSystemManager.h"

class BasicTutorial_00 : public BaseApplication
{
public:
	BasicTutorial_00(void);
	virtual bool frameStarted(const Ogre::FrameEvent& evt);
	virtual void createViewports(void);
	virtual void createScene(void);
	virtual void createCamera(void);
	virtual void chooseSceneManager(void);

	virtual bool keyPressed( const OIS::KeyEvent &arg );
	virtual bool keyReleased( const OIS::KeyEvent &arg );
	virtual bool mouseMoved( const OIS::MouseEvent &arg );
	
    //
    virtual void createWaterSurface();
    virtual void createLights();
    virtual void createParticleSystems();
    virtual void createLargeSphere();
    virtual void createMapMesh();
    virtual void createAvatar();
    virtual void createMonsterManager();
    virtual void createStatusBars();
	
private:
	/*!
	\brief Create a viewport

	Create a viewport for the entire screen.

	\return The sum of two integers.
	*/
	void createViewport_00(void);
	void createViewport_01(void);
	//
	void createCamera_00();
	void createCamera_01();

	void createScene_00();
	void createScene_01();
    
    SIMPLE_TERRAIN *mMapMesh;

	bool mFlgMotion;
	bool mKeyPressed;
	Ogre::Real mToggle;
	Ogre::Light *mLight0;

    Ogre::SceneNode* mParticleMainNode;

    int mNumParticleNodes;
	Ogre::SceneNode* mParticleNode[16];

	Ogre::SceneNode *mSN_Sphere;
	Ogre::Camera* mCameraArr[8];
	Ogre::SceneManager* mSceneMgrArr[8];
	OgreBites::SdkCameraMan* mCameraManArr[8];
//
	Real mSphereRadius;
	int mKeyPressedZoomMode;
	Real mCameraDistanceSlowDownSpeed;
	Real mCameraDistance;
	Real mCameraDistanceAdjustSpeed;
	Real mCameraV;
	MAIN_CHAR *mMainChar;
	MONSTER_MANAGER *mMonsterMgr;
    //
    float mEnergy;
    float mEnergy_Min;
    float mEnergy_Max;

    BAR_2D *mBar2D_Energy;
BAR_2D *mBar2D_2_Speed;

float mScoreCoord_X;
float mScoreCoord_MaxX;
float mScoreCoord_MinX;
bool mScoreBoard_Direction;
float mfScore;
int mScore;
	DIGIT_STRING_DIALOGUE *mDigitDialogue;
    //
    int mLevel;
	DIGIT_STRING_DIALOGUE *mDigitDialogue_Level;
    //
    WeaponParticleSystemManager *mWeaponPSMgr;
    //
    bool mTurnOnParticleSystems;
};

#endif // #ifndef __BasicTutorial_00_h_