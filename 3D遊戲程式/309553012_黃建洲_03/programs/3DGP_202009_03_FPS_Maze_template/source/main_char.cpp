#include "main_char.h"
#include "BasicTools.h"
#include "envTools.h"
#include "map_manager.h"
#include "sound_manager.h"

#include <OgreInstancedGeometry.h>

MAIN_CHAR::MAIN_CHAR()
{
mDistanceOffsetToTerrain = 0;
	mCamera = 0;
	mVelocity = Vector3(1, 0, 0);
mSpeedFactor = 100.0;
mActionMode = ACTION_NONE;
mEyePosition = Vector3(0,120, 0);

mFireActionMode = FIRE_ACTION_NONE;
mWeaponMgr = 0;
mTarget = 0; //null
mCurBulletsNum = 0;
}

MAIN_CHAR::MAIN_CHAR(SceneManager *a_SceneMgr) : GAME_OBJ(a_SceneMgr)
{
    mDistanceOffsetToTerrain = 0;
	mCamera = 0;
	mVelocity = Vector3(1, 0, 0);
mSpeedFactor = 100.0;
mActionMode = ACTION_NONE;
mEyePosition = Vector3(0,120, 0);

mFireActionMode = FIRE_ACTION_NONE;

mWeaponMgr = new WEAPON_MANAGER(mSceneMgr);

mTarget = 0; //null
mCurBulletsNum = 0;
}

void MAIN_CHAR::installWeaponWSManager(WeaponParticleSystemManager *wpsMgr)
{
    if (mWeaponMgr) mWeaponMgr->installWeaponWSManager(wpsMgr);
}

void MAIN_CHAR::setEyePosition_Y(double y)
{
    mEyePosition.y = y;
}

void MAIN_CHAR::setWalkingMaxSpeed_Modifier(double walkingMaxSpeed)
{
    mSpeedFactor_Modifer = walkingMaxSpeed;
}


WEAPON_MANAGER *MAIN_CHAR::getWeaponManager( )
{
	return mWeaponMgr;
}

void MAIN_CHAR::setMaxBulletsNum(int a_Num)
{
	mCurBulletsNum = a_Num;
	mWeaponMgr->setMaxBulletsNum(mCurBulletsNum);
}

void MAIN_CHAR::attachCamera(Camera *a_Camera)
{
	mCamera = a_Camera;
	FrameEvent evt;
	evt.timeSinceLastFrame = 0;
	walkForward(evt);
}

void MAIN_CHAR::updateViewDirection()
{
	Vector3 actualDirection;
	actualDirection = mCamera->getRealDirection();
	Vector3 robotDirection = mSceneNode->getOrientation() * Vector3::UNIT_Z;

	
	robotDirection.normalise();
	robotDirection.y = 0;
	actualDirection.normalise();
	actualDirection.y = 0;
	Ogre::Quaternion quat = robotDirection.getRotationTo(actualDirection);
	mSceneNode->yaw(Ogre::Degree(-90));
	
    mSceneNode->rotate(quat);
	

    // add your own stuff
}

void MAIN_CHAR::walkForward(const Ogre::FrameEvent& evt)
{
	Vector3 actualDirection = mCamera->getRealDirection();
	actualDirection.y = 0;
	
    //Vector3 actualDirection = Vector3(1, 0, 0);

    Vector3 d;
	d = actualDirection*mSpeedFactor*evt.timeSinceLastFrame
        *mSpeedFactor_Modifer;

	//logMessage("Direction\n");
	//logMessage(actualDirection);

	//logMessage(d);
	mSceneNode->translate(d);

	Vector3 pos = mSceneNode->getPosition();
	bool flg = projectScenePointOntoTerrain_PosDirection(pos);
	if (flg == false) {
		projectScenePointOntoTerrain_NegDirection(pos);
	}
	mSceneNode->setPosition(pos);
}

void MAIN_CHAR::walkBackward(const Ogre::FrameEvent& evt)
{
    // Add your own stuff
	Vector3 actualDirection = mCamera->getRealDirection() * -1;
	actualDirection.y = 0;
    Vector3 d;
	d = actualDirection*mSpeedFactor*evt.timeSinceLastFrame
        *mSpeedFactor_Modifer;

	//logMessage("Direction\n");
	//logMessage(actualDirection);

	//logMessage(d);
	mSceneNode->translate(d);

	Vector3 pos = mSceneNode->getPosition();
	bool flg = projectScenePointOntoTerrain_PosDirection(pos);
	if (flg == false) {
		projectScenePointOntoTerrain_NegDirection(pos);
	}
	mSceneNode->setPosition(pos);
}

void MAIN_CHAR::setPosition_to_Environment(const Vector3 &p)
{
	//mActualPos = p;
	mSceneNode->setPosition(p);
	Vector3 new_p;
	Vector3 cur_p = mSceneNode->getPosition();
	clampToEnvironment(cur_p, mDistanceOffsetToTerrain, new_p);
	mSceneNode->setPosition(new_p);
	//if (mModelNode) {
	//	mModelNode->setPosition(new_p);
	//}
}

unsigned int MAIN_CHAR::getActionMode() const {
    return mActionMode;
}

void MAIN_CHAR::setWalkForward()
{
	mActionMode |= ACTION_WALK_FORWARD;
}
	void MAIN_CHAR::setWalkBackward()
	{
	mActionMode |= ACTION_WALK_BACKWARD;

	}

	void MAIN_CHAR::unsetWalkForward()
{
	mActionMode ^= ACTION_WALK_FORWARD;
}
	void MAIN_CHAR::unsetWalkBackward()
	{
	mActionMode ^= ACTION_WALK_BACKWARD;

	}

	Vector3 MAIN_CHAR::getWeaponPosition() const
	{
		Vector3 p = mSceneNode->getPosition();
		p += mEyePosition;
		Vector3 d = mCamera->getRealDirection();
		p += d*20;
		return p;
	}

	void MAIN_CHAR::update(const Ogre::FrameEvent& evt)
	{
        Vector3 p0 = mSceneNode->getPosition();
        ///////////////////////////////////////////
		if (mActionMode & ACTION_WALK_FORWARD) {
			walkForward(evt);
		}
				if (mActionMode & ACTION_WALK_BACKWARD) {
			walkBackward(evt);
		}
		
		fireWeapon();
		updateWeapon(evt);
		Real sf = 3.0;
		//if (mAnimationState == 0) {
			if (
				(mActionMode & ACTION_WALK_FORWARD)
				||
				(mActionMode & ACTION_WALK_BACKWARD)
				) {
					mAnimationState = mEntity->getAnimationState("Walk");
	mAnimationState->setLoop(true);
	mAnimationState->setEnabled(true);
	if (mActionMode & ACTION_WALK_FORWARD) {
	mAnimationState->addTime(evt.timeSinceLastFrame*sf);
	}
	else {
		mAnimationState->addTime(-evt.timeSinceLastFrame*sf);
	}
	} else {
				mAnimationState = mEntity->getAnimationState("Idle");
	mAnimationState->setLoop(true);
	mAnimationState->setEnabled(true);
	mAnimationState->addTime(evt.timeSinceLastFrame*sf);
			}
		//}

Vector3 new_p;
	Vector3 cur_p = mSceneNode->getPosition();
    Vector3 modified_p;
	MAP_MANAGER::movePosition(p0, cur_p, modified_p);
	//
	clampToEnvironment(modified_p, 0.1, new_p);
    mSceneNode->setPosition(new_p);

    //////////////////////////////////////////////
    //Make the camera follow the main character.
    Vector3 pos = new_p + mEyePosition;
    Vector3 actualDirection = mCamera->getDirection();

    mCamera->setPosition(pos-actualDirection*5);
	mCamera->lookAt(pos);

	}

	void MAIN_CHAR::fireWeapon()
	{
		Vector3 pos;
		Vector3 direction;
		if (mFireActionMode&FIRE_ACTION_NORMAL)
		{
			
			pos = getWeaponPosition();
			direction = mCamera->getRealDirection();
			
			mWeaponMgr->fire_Normal(pos, direction);
		    mFireActionMode ^= FIRE_ACTION_NORMAL;
		}
	}

	void MAIN_CHAR::updateWeapon(const Ogre::FrameEvent& evt)
	{
		mWeaponMgr->update(evt);
	}

    // 
    // Many things can be done here....
    // For examples:
    // Play sound
    // Level up
    //
	void MAIN_CHAR::setFireAction_Normal()
	{
        //
        static int count = 0;
        count++;
        if (count%10==0) {
			SOUND_MANAGER::getInstance()->play_LevelUp();
            ++mLevel;
        }
        //
		mFireActionMode |= FIRE_ACTION_NORMAL;
	}