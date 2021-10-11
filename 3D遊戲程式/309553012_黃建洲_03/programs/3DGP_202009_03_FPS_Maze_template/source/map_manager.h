/*
This is a game demo written by Wingo Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/

#ifndef __MAP_MANAGER_H_
#define __MAP_MANAGER_H_
#include "Ogre.h"
#include "OgreStringConverter.h"
#include "OgreException.h"
#include "OgreMaterialManager.h"
#include "OgreOverlayManager.h"
#include "OgreTechnique.h"
#include "OgreBlendMode.h"
#include "OgreOverlay.h"
#include "game_obj.h"
#include "wago_ogre_mesh.h"

class MAP_MANAGER {
private:
	static GAME_OBJ **mObj;
	static int mNumObj;
	static int mMaxNumObj;
	static SIMPLE_TERRAIN *mMeshMapMgr;
	static void computePositionBasedOnMeshMapMgr(const Vector3 &p0, const Vector3 &p1, Vector3 &p);
protected:
public:
	MAP_MANAGER();
	static void addObstacle(GAME_OBJ *a_GameObj);
    static bool movePosition_Obstacles(const Vector3 &p0, const Vector3 &p1, Vector3 &p);
	static bool movePosition(const Vector3 &p0, const Vector3 &p1, Vector3 &p);
	static Vector3 getGridNormalVector( float x, float z);
	static bool inside(const Vector3 &p0, const Vector3 &p1, Real t, const Plane *halfPlanes, int numHalfPlanes);
	static void installMeshMapManager(SIMPLE_TERRAIN *a_MeshMgr);
};
#endif