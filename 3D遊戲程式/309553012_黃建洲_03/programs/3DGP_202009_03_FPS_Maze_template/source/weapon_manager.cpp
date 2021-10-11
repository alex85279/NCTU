/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/
#include "weapon_manager.h"

WEAPON_MANAGER::WEAPON_MANAGER(SceneManager *a_SceneMgr) : GAME_OBJ(a_SceneMgr)
{
	mCoolDownTimeForFiring = 0.0;
	mMaxCoolDownTime = 0.1;
	//
	mMaxNum = 128;
	mCurBulletsNum = mMaxNum;
	for (int i = 0; i < mMaxNum; ++i) {
		mUsedWeaponsArr[i] = false;
		 mBulletsArr[i] = new WEAPON(mSceneMgr);
		 mBulletsArr[i]->createGameObj("b", "sphere.mesh");
		mBulletsArr[i]->scale(0.05, 0.05, 0.05);
		mBulletsArr[i]->setVisible(false);
		mBulletsArr[i]->getEntity( )->setMaterialName("Examples/RustySteel");
	}
	mFlgTarget = false;
	//
	mWeaponPSMgr = 0;
}

void WEAPON_MANAGER::setTarget( const Vector3 &pos, Real radius )
{
		mTargetPos = pos;
		mTargetRadius = radius;
	mFlgTarget = true;
}


void WEAPON_MANAGER::setMaxBulletsNum(int a_Num)
{
	//std::cout<<a_Num<<std::endl;
	if (a_Num >= mMaxNum) {
		a_Num = mMaxNum;
	}

	mCurBulletsNum = a_Num;
}

//
// Find a free weapon to use
// pos: initial position of the weapon
// direction: the shooting direction
//
void WEAPON_MANAGER::fire_Normal(const Vector3 &pos, const Vector3 &direction)
{
	mCurBulletsNum = 128;
	GAME_OBJ *g = 0;
	for (int i = 0; i < mCurBulletsNum; ++i) {
		// Check if a weapon is not used
		if (mUsedWeaponsArr[i] == true) continue;
		g = mBulletsArr[i];
		mUsedWeaponsArr[i] = true;
		break;
	}
	if (g == 0) return;
	if (mCoolDownTimeForFiring < mMaxCoolDownTime) return;
	mCoolDownTimeForFiring = 0;

	//
	// Add your own stuff
	// Set the weapn velocity properly here
	// Use g->setVelocity

	g->setPosition(pos);
	// Set velocity here
	g->setVelocity(direction*200);
	g->setSpeedFactor(1);
	g->setLife(5, 5);
	g->makeAlive();

}

void WEAPON_MANAGER::installWeaponWSManager(WeaponParticleSystemManager *wpsMgr)
{
    mWeaponPSMgr = wpsMgr;
}

void WEAPON_MANAGER::update(const Ogre::FrameEvent& evt)
{
	mCoolDownTimeForFiring += evt.timeSinceLastFrame;
	if (mCoolDownTimeForFiring > mMaxCoolDownTime)
	{
		mCoolDownTimeForFiring = mMaxCoolDownTime;
	}
	for (int i = 0; i < mMaxNum; ++i) {
		if (mUsedWeaponsArr[i] == false) continue;
		WEAPON *g = mBulletsArr[i];
		g->update(evt);

		if ( mFlgTarget ) {
            if (!mWeaponPSMgr) {
                g->hitTarget_Sphere( 
                mTargetPos,
                mTargetRadius 
                );
				
            } else {
			g->hitTarget_Sphere( 
                mTargetPos, 
                mTargetRadius,
                mWeaponPSMgr
                );
            }
			g->adjustDueToMap( );
		}
		if (!g->isAlive()) {
			mUsedWeaponsArr[i] = false;
		}
	}

}
