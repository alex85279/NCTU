#include "game_obj.h"
#include "BasicTools.h"
//
int GAME_OBJ::gs_NameIndex = 0;
GAME_OBJ::GAME_OBJ(SceneManager *a_SceneMgr)
{
	mSceneMgr = a_SceneMgr;
	mRadius = 10;
	mMass = 1.0;
	mVelocity = Vector3(0, 0, 0);
	mSpeedFactor = 1.0;
	mInitDirection = Vector3(1, 0, 0);
	//
	mMaxLifeTime = 10;
	mLifeTime = mMaxLifeTime;
	mIsAlive = false;
	mTime = 0;
	mRandSpeed = (rand()%100)/100.0;
	mAmplitude = 20;
	mAnimationState = 0;
    //
    mLevel = 1;
}

void GAME_OBJ::setObjSpace(Plane *a_Plane, int a_Num)
{
	mNumHalfPlanes = a_Num;
	for (int i =0; i < mNumHalfPlanes; i++) {
		mHalfPlane[i] = a_Plane[i];
	}
}

void GAME_OBJ::translate(const Vector3 &v)
{
	mSceneNode->translate(v);
}

void GAME_OBJ::setTarget(GAME_OBJ *a_Target, Real radius)
{
	mTarget = a_Target;
	mTargetRadius = radius;
}

void GAME_OBJ::setVisible(bool flg)
{
	mSceneNode->setVisible(flg);
}

void GAME_OBJ::setVisibilityFlags(unsigned int m)
{
	mEntity->setVisibilityFlags(m);
}

void GAME_OBJ::makeAlive(bool flg)
{
	mIsAlive = flg;
	if (flg) {
		mSceneNode->setVisible(true);
	} else {
		mSceneNode->setVisible(false);
	}
}


void GAME_OBJ::setLife(Real cLife, Real cMaxLife)
{
	mLifeTime = cLife;
if (cMaxLife > 0) mMaxLifeTime = cMaxLife;
}

bool GAME_OBJ::isAlive() const
{
	return mIsAlive;
}


void GAME_OBJ::update(const Ogre::FrameEvent& evt)
{

}

void GAME_OBJ::createGameObj(const String &a_Name, const String &a_MeshName)
{
	String name;
	genNameUsingIndex(a_Name, gs_NameIndex, name);

	mEntity = mSceneMgr->createEntity(name, a_MeshName);
	mSceneNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	mSceneNode->attachObject(mEntity);

	++gs_NameIndex;
}

void GAME_OBJ::scale(Real sx, Real sy, Real sz)
{
	mSceneNode->scale(sx, sy, sz);
}


void GAME_OBJ::setSpeedFactor(Real f)
{
	mSpeedFactor = f;
}

void GAME_OBJ::setPosition(const Vector3 &pos)
{
	mSceneNode->setPosition(pos);
}

void GAME_OBJ::setInitPosition(const Vector3 &pos)
{
	mInitPosition = pos;
}

void GAME_OBJ::setVelocity(const Vector3 &v)
{
	mVelocity = v;
}
