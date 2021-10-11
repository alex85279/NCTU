/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#ifndef __GAME_OBJ_NAME_FRACTORY_H__
#define __GAME_OBJ_NAME_FRACTORY_H__
#include "ogre.h"

class GAME_OBJ_NAME_FACTORY {
protected:
	static Ogre::String global_name_prefix;
	static int global_name_counter;
	static void generateGlobalObjName(Ogre::String &name) {
		global_name_counter++;
		name = global_name_prefix + Ogre::StringConverter::toString(static_cast<int>(global_name_counter));
	}
public:
	GAME_OBJ_NAME_FACTORY(){}
};

#endif