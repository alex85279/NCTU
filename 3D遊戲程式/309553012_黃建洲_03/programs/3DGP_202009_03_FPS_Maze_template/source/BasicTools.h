#ifndef __BASIC_TOOLS_H__
#define __BASIC_TOOLS_H__
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>

#include <OISEvents.h>
#include <OISInputManager.h>
#include <OISKeyboard.h>
#include <OISMouse.h>

#include <SdkTrays.h>
#include <SdkCameraMan.h>

extern void genNameUsingIndex(const Ogre::String & prefix, int index, Ogre::String &out_name);
extern void logMessage(const  Ogre::String &msg);
extern void logMessage(const Ogre::Vector3 &v);

extern void bt_Init(OgreBites::SdkTrayManager* a_TrayMgr, Ogre::SceneManager *a_SceneMgr, Ogre::Camera *a_Camera);
extern bool projectScenePointOntoTerrain_PosDirection(Ogre::Vector3 &p);
extern bool projectScenePointOntoTerrain_NegDirection(Ogre::Vector3 &p);

extern Ogre::SceneNode* initParticleSystemForExplosion(
    Ogre::SceneManager *sceneMgr,
    const char *particleName
    );
extern void setOffParticleSystem(
	Ogre::SceneNode *fNode, const Ogre::String &pname, const Ogre::Vector3 &pos); 
extern void setVisibleParticleSystem(
	Ogre::SceneNode *particleNode, 
    const Ogre::String &pname,
    bool flg_show); 

#endif