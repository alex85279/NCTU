//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
// Date: Nov 2020
//
#include "TutorialApplication.h"
#include "BasicTools.h"
#include "read_data.h"
#include "map_manager.h"
#include "envTools.h"
#include "sound_manager.h"

using namespace Ogre;
SOUND *mSound;

BasicTutorial_00::BasicTutorial_00(void) {
	mFlgMotion = true;
	mToggle = 2.5;
	mKeyPressed = false;
	//
	mMainChar = 0;
	mCameraDistance = 0;
	mKeyPressedZoomMode = 0;
mCameraDistanceAdjustSpeed = 0;
mCameraDistanceSlowDownSpeed = 0;
//mSound = new SOUND;
//mSound->init();
//
mScoreCoord_MaxX = 0.5;
 mScoreCoord_MinX = 0.05;
mScoreCoord_X = mScoreCoord_MinX;

 mScoreBoard_Direction = true;
 mfScore = 0.0;

 mEnergy_Min = 0.0;
 mEnergy_Max = 100.0;
 mEnergy = mEnergy_Max;

 mTurnOnParticleSystems= true;
 //
 mLevel = 1;
}

bool BasicTutorial_00::mouseMoved( const OIS::MouseEvent &arg )
{
	bool flg = BaseApplication::mouseMoved(arg);
	mMainChar->updateViewDirection();
	return flg;
}

bool BasicTutorial_00::keyPressed( const OIS::KeyEvent &arg )
{
	//bool flg = BaseApplication::keyPressed(arg);
	bool flg = true;
	
    if ( arg.key == OIS::KC_P) {
        mTurnOnParticleSystems = !mTurnOnParticleSystems;

        for (int i = 0; i < mNumParticleNodes; ++i) {
            setVisibleParticleSystem(
                mParticleNode[i], 
                "explosion", 
                mTurnOnParticleSystems);
        }

    }
	if (arg.key == OIS::KC_F1) {
		mMainChar->setPosition_to_Environment(Vector3(860,51,751));
	}
	
	if (arg.key == OIS::KC_F2) {
		mMainChar->setPosition_to_Environment(Vector3(4612,41,4529));
	}

	if (arg.key == OIS::KC_SPACE)
	{
		mMainChar->setFireAction_Normal();
	}

	if (arg.key == OIS::KC_W) {
		mMainChar->setWalkForward();
	}
	if (arg.key == OIS::KC_S) {
		mMainChar->setWalkBackward();
	}

	
	if (arg.key == OIS::KC_SPACE) {
		//mKeyPressed = true;
	}

	if (arg.key == OIS::KC_Z) {
		mKeyPressedZoomMode= 1;
		mCameraDistanceAdjustSpeed = 300;
		mCameraDistanceSlowDownSpeed = 0;
	}
	if (arg.key == OIS::KC_X) {
		mKeyPressedZoomMode = -1;
		mCameraDistanceAdjustSpeed = 300;
		mCameraDistanceSlowDownSpeed = 0;

	}
	return flg;
}

bool BasicTutorial_00::keyReleased( const OIS::KeyEvent &arg )
{
	bool flg = true;
	if (arg.key == OIS::KC_Z) {
		mKeyPressedZoomMode= 0;
		mCameraDistanceAdjustSpeed = -mCameraDistanceAdjustSpeed;
		mCameraDistanceSlowDownSpeed = -200;
	}
	if (arg.key == OIS::KC_X) {
		mKeyPressedZoomMode = 0;
		mCameraDistanceAdjustSpeed = mCameraDistanceAdjustSpeed;
		mCameraDistanceSlowDownSpeed = 200;
	}

	if (arg.key == OIS::KC_W) {
		mMainChar->unsetWalkForward();
	}
	if (arg.key == OIS::KC_S) {
		mMainChar->unsetWalkBackward();
	}

	if (arg.key == OIS::KC_ESCAPE)
	{
		mShutDown = true;
	}
	return flg;
}
void BasicTutorial_00::chooseSceneManager()
{
	//ST_EXTERIOR_FAR
	mSceneMgrArr[0] = mRoot
		->createSceneManager(ST_EXTERIOR_CLOSE, "primary");
	//mSceneMgrArr[0] = mRoot
	//	->createSceneManager(
	//ST_EXTERIOR_CLOSE, "primary");

    /*mSceneMgrArr[0] = mRoot
		->createSceneManager(
	ST_GENERIC, "primary");

	mSceneMgrArr[1] =mSceneMgrArr[0];*/
	/*
	mSceneMgrArr[1] = mRoot
	->createSceneManager(ST_EXTERIOR_CLOSE, "secondary");
*/
	}

void BasicTutorial_00::createCamera_00(void)
{
	mSceneMgr = mSceneMgrArr[0];
	mCamera = mCameraArr[0] = mSceneMgr->createCamera("PlayerCam");
	mCamera->setPosition(Ogre::Vector3(770,800,2100));
	mCamera->lookAt(Ogre::Vector3(750,50,750));
	mCamera->setNearClipDistance(5);
	mCameraManArr[0] = new OgreBites::SdkCameraMan(mCamera);   // create a default camera controller
}

void BasicTutorial_00::createCamera_01(void)
{
	mSceneMgr = mSceneMgrArr[0];
	mCamera = mCameraArr[1] = mSceneMgr->createCamera("PlayerCam2");
	mCamera->setPosition(Ogre::Vector3(750,1000,750));
	mCamera->lookAt(Ogre::Vector3(750, 0, 750.1));
	mCamera->setNearClipDistance(5);
	mCameraManArr[1] = new OgreBites::SdkCameraMan(mCamera);   // create a default camera controller

}



void BasicTutorial_00::createViewport_00(void)
{
	mCamera = mCameraArr[0];
	Ogre::Viewport* vp = mWindow->addViewport(mCamera, 0, 0, 0);
	vp->setBackgroundColour(Ogre::ColourValue(0,0,1));
	mCamera->setAspectRatio(
		Ogre::Real(vp->getActualWidth()) / Ogre::Real(vp->getActualHeight()));
	vp->setVisibilityMask(0x01);
}

void BasicTutorial_00::createViewport_01(void)
{
	mCamera = mCameraArr[1];
	Ogre::Viewport* vp = mWindow->addViewport(mCamera, 1, 0.75, 0.0, 0.25, 0.25);
	vp->setBackgroundColour(Ogre::ColourValue(0,0,1));
	mCamera->setAspectRatio(
		Ogre::Real(vp->getActualWidth()) / Ogre::Real(vp->getActualHeight()));
	vp->setOverlaysEnabled(false);
    vp->setVisibilityMask(0x10);
//	vp->setVisibilityMask(0x10);
}


void BasicTutorial_00::createWaterSurface()
{
    	Plane plane(Vector3::UNIT_Y, 0); 
	MeshManager::getSingleton().createPlane(
		"ground", 						ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		plane, 
		15000,15000, 	// width, height
		20,20, 		// x- and y-segments
		true, 		// normal
		1, 			// num texture sets
		5,5, 		// x- and y-tiles
		Vector3::UNIT_Z	// upward vector
		); 

	Entity *ent = mSceneMgr->createEntity(
		"GroundEntity", "ground"); 
	//ent->setMaterialName("Examples/BeachStones");
	//ent->setMaterialName("Examples/WaterStream");

	ent->setMaterialName(DATA_READER::getWaterMaterialName());
	Vector3 waterCoordY;
	waterCoordY.y = DATA_READER::getWaterCoord_Y();

	SceneNode* node = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	node->attachObject(ent);
    node->setPosition(waterCoordY);
    //
    // Add your own stuff to create a large plane
    //
}

void BasicTutorial_00::createLights()
    {
    Light *light;
	light = mSceneMgr->createLight("Light1"); 
	light->setType(Light::LT_SPOTLIGHT);
	light->setPosition(Vector3(750, 1000, 750));
	light->setDiffuseColour(1, 1, 1);		
	light->setSpecularColour(1.0, 1.0, 1.0);
	light->setDirection(Vector3(0, -1, 0));
	light -> setSpotlightRange(Degree(10), Degree(60), 1.0);
	mLight0 = light;


	Light *light2;
	light2 = mSceneMgr->createLight("Light2"); 
	light2->setType(Light::LT_POINT);
	light2->setPosition(Vector3(750, 1000, 750));
	light2->setDiffuseColour(1, 1, 1);		
	light2->setSpecularColour(1.0, 1.0, 1.0);
	light2->setDirection(Vector3(0, -1, 0));

}

void BasicTutorial_00::createParticleSystems()
{
    mParticleMainNode = initParticleSystemForExplosion(
        mSceneMgr,
        "explosion");
	Vector3 pos;
	setOffParticleSystem(mParticleMainNode, "explosion", pos);

    mNumParticleNodes = 16;
    for (int i = 0; i < mNumParticleNodes; ++i) {
        String pname;
        genNameUsingIndex("p", i, pname);
	    mParticleNode[i] = initParticleSystemForExplosion(
            mSceneMgr,
            pname.data());
        mParticleNode[i]->setVisible(false);
    }

}

void BasicTutorial_00::createLargeSphere()
{
     Vector3 org = Vector3(750, 51, 750);

    	Entity *ent2 
		= mSceneMgr
		->createEntity( "Robot2", "sphere.mesh" ); 
	//ent2->setCastShadows(true);

	
	
	ent2->setMaterialName("Examples/SphereMappedRustySteel");
	SceneNode *node2 = mSN_Sphere
		= mSceneMgr
		->getRootSceneNode()
		->createChildSceneNode( 
		"RobotNode2", org+Vector3( 0, 50, 0 ) ); 
	Real scalingFactor = 2.0;
	mSN_Sphere->scale(scalingFactor, scalingFactor, scalingFactor );
	mSphereRadius = ent2->getBoundingRadius( );
	mSphereRadius *= scalingFactor*0.7;
	node2->attachObject( ent2 );

}

void BasicTutorial_00::createMapMesh()
{
    mMapMesh = new SIMPLE_TERRAIN(mSceneMgr);
		mMapMesh->setMaterial("Examples/Rockwall_Wingo");
		mMapMesh->create();
		//mMapMesh->translateSceneNode(10000, -300, 10000);
		mMapMesh->translateSceneNode(0, -300, 0);
		mMapMesh->dilateMapObstacles(2);
        mMapMesh->computeNormalVectors( );
		mMapMesh->scanMapForLocatingObjects();

        MAP_MANAGER::installMeshMapManager(mMapMesh);
}

void BasicTutorial_00::createAvatar()
{
    mMainChar = new MAIN_CHAR(mSceneMgrArr[0]);
	//mMainChar->createGameObj("m", "robot.mesh");
    mMainChar->createGameObj("avatar", DATA_READER::getAvatarMeshName());
    mMainChar->setEyePosition_Y(DATA_READER::getAvatarEyePosition_Y());

    mMainChar->setVisibilityFlags(0x10);

	mMainChar->attachCamera(mCameraArr[0]);

	mMainChar->setMaxBulletsNum(DATA_READER::getMaxBulletsNum());
    mMainChar->setWalkingMaxSpeed_Modifier(DATA_READER::getAvatarWalkingMaxSpeed());

    Vector3 start_pos;
		mMapMesh->getStartingPosition(start_pos);
		//mMainChar->setPosition(start_pos);
        mMainChar->setPosition_to_Environment(start_pos);

        mMainChar->installWeaponWSManager(mWeaponPSMgr);
}

void BasicTutorial_00::createMonsterManager()
{
    mMonsterMgr = new MONSTER_MANAGER(mSceneMgrArr[0]);

mMonsterMgr->setTargetForMonsters(mMainChar);
mMonsterMgr->setMaxMonstersNum(DATA_READER::getMaxMonstersNum());
}

void BasicTutorial_00::createStatusBars() {
    mBar2D_Energy = new BAR_2D(mSceneMgr, "wago_game/StaminaBar", NULL);
	mBar2D_Energy->setDefaultPos(0.05, 0.55);

    mBar2D_2_Speed = new BAR_2D(mSceneMgr, "wago_game/LifeBar", NULL);
	mBar2D_2_Speed->setDefaultPos(0.05, 0.65);
//
    //wago_game/ManaBar

    //
    mScore = 123;
    mDigitDialogue = new DIGIT_STRING_DIALOGUE(mSceneMgr, "Examples/Digits");
	
	mDigitDialogue->setScore(mScore, 0.05, 0.05);

    //mLevel = 123;
    	mDigitDialogue_Level = new DIGIT_STRING_DIALOGUE(mSceneMgr, "Examples/Digits");
	mDigitDialogue_Level->setScore(mLevel, 0.05, 0.1);
}
void BasicTutorial_00::createScene_00(void) 
{
    Ogre::LogManager::getSingletonPtr()->logMessage("*** createScene_00 ***");
    Ogre::LogManager::getSingletonPtr()->logMessage("*** DATA_READER::readData() ***");

	DATA_READER::readData();
	bt_Init(mTrayMgr, mSceneMgrArr[0], mCameraArr[0]);
    //
    initEnvTools(mSceneMgrArr[0]);
    //
	mSceneMgr = mSceneMgrArr[0];
	//
    mWeaponPSMgr = new WeaponParticleSystemManager(mSceneMgr);
    mWeaponPSMgr->init();

    //
	ColourValue fadeColour(0.9, 0.9, 0.9); 

    //
    // Load a terrain here
    //
	mSceneMgr->setWorldGeometry("terrain.cfg"); 
	/*
	This line should put before SetWorldGeometry
	*/

    //
    // Enable fog?
    //
	if (DATA_READER::isEnabledExpFog()) {
		Real density = DATA_READER::getExpFogDensity();
		mSceneMgr->setFog(FOG_EXP, fadeColour, density); 
	}

    //
    // Create a skybox here
    //
	mSceneMgr->setSkyBox(
		true, 
		"Examples/SpaceSkyBox");
	if (DATA_READER::isEnabledShadow()) {
		mSceneMgr->setShadowTechnique(
			SHADOWTYPE_TEXTURE_MODULATIVE);
			//SHADOWTYPE_TEXTURE_ADDITIVE);
		//SHADOWTYPE_STENCIL_MODULATIVE);
	}

	// 
	
//mSceneMgr->setAmbientLight( ColourValue( 0.7, 0.7, 0.7 ) ); 
	mSceneMgr->setAmbientLight( ColourValue( 1, 1, 1 ) );  

    Ogre::LogManager::getSingletonPtr()->logMessage("*** DATA_READER::createLights() ***");

createLights();

createParticleSystems();

createLargeSphere();
	
	//

    createWaterSurface();

    createMapMesh();
    //
    		
    //
    createAvatar();
	
    //
    createMonsterManager();

    createStatusBars();
//
    
    SOUND_MANAGER::getInstance();
    //
    
    Ogre::LogManager::getSingletonPtr()->logMessage("*** DATA_READER::createStatusBars() ***");

    
	//

}

void BasicTutorial_00::createScene_01(void) 
{
	return;
	mSceneMgr = mSceneMgrArr[1];
	mSceneMgr->setAmbientLight( ColourValue( 1.0,1.0, 1.0 ) ); 
	//mSceneMgr->setAmbientLight( ColourValue( 0.5,0.5, 0.5 ) );  
	

		mSceneMgr->setWorldGeometry("terrain.cfg"); 


	
}

//
// Do game logic here
//
// e.g., perform animation update, 
// setting lights' positions and so on
//
bool BasicTutorial_00::frameStarted(const Ogre::FrameEvent& evt)
{
	bool flg = Ogre::FrameListener::frameStarted(evt);

	mToggle += evt.timeSinceLastFrame;
	if (mToggle > 2.0) mToggle = 2.5;

	if (mKeyPressed) {

		if (mToggle >= 2.0) {
			mFlgMotion = !mFlgMotion;
			mToggle = 0.0;
		}
	}

	mKeyPressed = false;
	if (mFlgMotion == false) return flg;

	Vector3 org = Vector3(750, 51, 750);
	Vector3 pos = Vector3(750, 330, 750);
	Real dt = evt.timeSinceLastFrame;

	mSN_Sphere->setPosition(pos);

    mMainChar->getWeaponManager()->setTarget( pos, mSphereRadius );
	
	Vector3 cpos = mMainChar->getPosition();
	
    
	if(mKeyPressedZoomMode == 1 || mKeyPressedZoomMode == -1){
		mCameraV = mKeyPressedZoomMode * mCameraDistanceAdjustSpeed * dt;
	}
	else{
		if(mCameraDistanceSlowDownSpeed > 0 && mCameraV < 0){
			mCameraV = mCameraV + mCameraDistanceSlowDownSpeed * dt * 0.01f;
			if(mCameraV > 0) mCameraV = 0;
		}
		else if(mCameraDistanceSlowDownSpeed < 0 && mCameraV > 0){
			mCameraV = mCameraV + mCameraDistanceSlowDownSpeed * dt * 0.01f;
			if(mCameraV < 0) mCameraV = 0;
		}
	}
	mCameraDistance = mCameraDistance - mCameraV;
    mCameraArr[1]->setPosition(cpos + Vector3(0, 1000 + mCameraDistance, 0));
	mLight0->setPosition(cpos + Vector3(0, 1000, 0));
	
	mMainChar->update(evt);
	
mMonsterMgr->update(evt);

if (mTurnOnParticleSystems) {
mMonsterMgr->setParticleSystem(
    mMainChar->getPosition(),
    mNumParticleNodes,
    mParticleNode
    );
}

{
    unsigned int mode =  mMainChar->getActionMode();
}

	//bar	
	unsigned int mode =  mMainChar->getActionMode();
	// Update the energy bar
	if ((mode & ACTION_WALK_FORWARD) || (mode & ACTION_WALK_BACKWARD)) 
	{ 
	    mEnergy -= 0.05;
		if (mEnergy < mEnergy_Min) {
			mEnergy = mEnergy_Min;
		}
	}
	else {
		mEnergy += 0.05;
        if (mEnergy > mEnergy_Max) {
		   mEnergy = mEnergy_Max;
	    }
	}

	float eRatio = mEnergy / mEnergy_Max;

	float eBarWidth = 0.25 * eRatio;
	mBar2D_Energy->setBarDimension(eBarWidth, 0.025);
	mBar2D_Energy->update(mCamera, 0.001, 0.05, 0.05);

	//speed bar
	float K = 200.0;
	float rSpeed = K * sqrt(eRatio);
	float sRatio = rSpeed / K;

	float sBarWidth = 0.25 * sRatio;
	mBar2D_2_Speed->setBarDimension(sBarWidth, 0.025);
	mBar2D_2_Speed->update(mCamera, 0.001, 0.05, 0.05);

    


	mMainChar->setSpeedFactor(rSpeed);
	

	//digit
	if(mScoreBoard_Direction == true){
		mScoreCoord_X += 0.001;
		if(mScoreCoord_X >= 0.5){
			mScoreBoard_Direction = false;
		}
	}
	else{
		mScoreCoord_X -= 0.002;
		if(mScoreCoord_X <= 0.05){
			mScoreBoard_Direction = true;
		}
	}
	mScore = mScore + 1;
	if(mScore >= 999){
		mScore = 123;
	}
	mDigitDialogue->setScore(mScore, mScoreCoord_X, 0.05);

	mLevel = mMainChar->getLevel();
	mDigitDialogue_Level->setScore(mLevel,0.05, 0.1);
	
	return flg;
}

void BasicTutorial_00::createViewports(void)
{
	createViewport_00();
	createViewport_01();
	mCamera = mCameraArr[0];
}

void BasicTutorial_00::createCamera(void) {
	createCamera_00();
	createCamera_01();
	mCameraMan = mCameraManArr[0];
	//mCameraMan = mCameraManArr[1];
	mCamera = mCameraArr[0];
}

void BasicTutorial_00::createScene( void ) {
	createScene_00();
	createScene_01();
	//mSceneMgr = mSceneMgrArr[0]; // active SceneManager
	mSceneMgr = mSceneMgrArr[1]; // active SceneManager
}



int main(int argc, char *argv[]) {
	BasicTutorial_00 app;
	app.go();  
	return 0;
}
