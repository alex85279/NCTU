/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2018
Email: wingo.wong@gmail.com

All rights reserved. 2009,2018
*/
#include "game_obj_name_factory.h"

//initialize static member variables
Ogre::String GAME_OBJ_NAME_FACTORY::global_name_prefix = "_ton_";
int GAME_OBJ_NAME_FACTORY::global_name_counter = 0;