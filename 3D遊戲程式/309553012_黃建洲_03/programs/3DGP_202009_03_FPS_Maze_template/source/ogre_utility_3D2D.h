/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - Dec 2020
Email: wingo.wong@gmail.com
*/
#ifndef __OGRE_UTILITY_H__
#define __OGRE_UTILITY_H__

#include "Ogre.h"
#include "OgreStringConverter.h"
#include "OgreException.h"

using namespace Ogre;

extern void mapWorldPositionToViewportCoordinates(const Camera *camera, const Vector3 &world_pos, Vector2 &viewport_coord);
#endif