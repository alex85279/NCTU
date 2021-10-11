/*
This is a game demo written by Wingo Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#ifndef __ENV_TOOLS_H__
#define __ENV_TOOLS_H__
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreRoot.h>
//#include "common.h"
//
using namespace Ogre;

#define ENABLE_COLLISION_DETECTION
//#undef ENABLE_COLLISION_DETECTION
//
#define DISABLE_COLLISION_DETECTION_WORLD
#undef DISABLE_COLLISION_DETECTION_WORLD

extern void initEnvTools(SceneManager *mSceneManager);
extern bool clampToEnvironment(const Vector3& cur_pos, Real offset, Vector3& new_pos);
//extern bool checkFireHitTarget_SceneObj(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance);
//extern bool checkFireHitTarget_World(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance);
//extern bool checkFireHitTarget(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance);
#endif