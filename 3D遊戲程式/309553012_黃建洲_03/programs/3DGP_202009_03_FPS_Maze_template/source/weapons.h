/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/
#ifndef __WEAPONS_H__
#define __WEAPONS_H__
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include "WeaponParticleSystemManager.h"
#include "game_obj.h"
//
class WEAPON : public GAME_OBJ {
protected:
public:
	WEAPON(SceneManager *a_SceneMgr);
	virtual void update(const Ogre::FrameEvent& evt);
	bool hitTarget_Sphere(const Vector3 &p, Real r );
    bool hitTarget_Sphere(
    const Vector3 &p, 
    Real r,
    WeaponParticleSystemManager *wpsMgr
    );

    void adjustDueToMap( );
};

#endif