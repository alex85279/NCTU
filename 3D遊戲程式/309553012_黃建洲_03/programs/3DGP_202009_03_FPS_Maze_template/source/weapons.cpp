/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/
#include "weapons.h"
#include "BasicTools.h"
#include "sound.h"
#include "sound_manager.h"
#include "map_manager.h"

extern SOUND *mSound;
WEAPON::WEAPON(SceneManager *a_SceneMgr) : GAME_OBJ(a_SceneMgr)
{
}

bool WEAPON::hitTarget_Sphere(
    const Vector3 &p, 
    Real r )
{
	bool flg = false;
	Vector3 pos = mSceneNode->getPosition();
	if ( r >= pos.distance( p ) ) {
		Vector3 normal_dir = pos - p;
		normal_dir.normalise( );
		Vector3 new_pos = p + (r+0.5)*normal_dir; 
		mSceneNode->setPosition( new_pos );
		Real d = mVelocity.dotProduct( normal_dir );
		if ( 0.0 >= d ) {
			mVelocity = mVelocity - d*normal_dir;
            mSound->init();
		    mSound->play();
            SOUND_MANAGER::getInstance()->play_Explosion();

        }
		flg = true;
	}
	return flg;
}

//
// p: sphere position
// r: radius
// wpsMgr: the particle system for the weapon
//
bool WEAPON::hitTarget_Sphere(
    const Vector3 &p, 
    Real r,
    WeaponParticleSystemManager *wpsMgr
    )
{
	bool flg = false;
	Vector3 pos = mSceneNode->getPosition();
	if ( r >= pos.distance( p ) ) {
        // the weapon overlaps with the sphere.
        // does this imply 'hit'?
        //
        // 
        //
        // Check if hitting...
		//if ( 0.0 >= d ) {
            // the weapon hits the target
            // Perform collision response
			// Compute a new mVelocity;
            // Play a sound?
            // Play a particle system?
            //
        //}
		Vector3 dir = pos - p;
		Vector3 normal_dir = dir.normalisedCopy();
		Vector3 n_pos = p + (r * 0.5) *	normal_dir;
		mSceneNode->setPosition(n_pos);
		Real d = mVelocity.dotProduct(normal_dir);
		if(d <= 0){
			mVelocity = mVelocity - d * normal_dir;
			SOUND_MANAGER::getInstance()->play_Explosion();
            wpsMgr->play(pos);
		
		}

		flg = true;
	}
	return flg;
}

void WEAPON::update(const Ogre::FrameEvent& evt)
{
		mLifeTime -= evt.timeSinceLastFrame;
	if (mLifeTime < 0 ) {
		mLifeTime = 0;
		mSceneNode->setVisible(false);
	mIsAlive = false;
	return;
	}
Real t = evt.timeSinceLastFrame;
	Vector3 pos = mSceneNode->getPosition();
	pos += mVelocity*t;

	Vector3 grav(0,-29.8, 0);
	mVelocity += grav*t;

	Vector3 new_pos = pos;
	Real r = 5;
	new_pos -= Vector3(0, 1, 0)*r;

    // Do collision check and collision response with the terrain.
    //
    // Add your own stuff
    //
	bool flg = projectScenePointOntoTerrain_PosDirection(new_pos);
    if (flg) {
        pos = new_pos + Vector3(0, 1, 0) + Vector3(0, 1, 0)*r;
        mVelocity.y = -mVelocity.y*0.9;
    }

mSceneNode->setPosition(pos);
}

//
// Adjust the position of the bullet when the bullet hits or is near to wall.
// Use MAP_MANAGER::getGridNormalVector(pos.x, pos.z ) to obtain the normal
// If the normal's length is zero, the bullet does not hit wall.
//
void WEAPON::adjustDueToMap( )
{
    Vector3 pos = mSceneNode->getPosition( );
	Vector3 n = MAP_MANAGER::getGridNormalVector(pos.x, pos.z );
        Real len = n.length( );
        if ( len!=0.0) {

			//pos = pos + n + n*5;
			mVelocity = mVelocity  - 2 * n * n.dotProduct(mVelocity);
            // Modify mVelocity based on the normal vector, n
			
            //  cout << "normal:" << n.x << "\t" << n.y << "\t" << n.z << endl;

        }
}