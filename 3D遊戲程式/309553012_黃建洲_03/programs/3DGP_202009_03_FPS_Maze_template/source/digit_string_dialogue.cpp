/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#include "digit_string_dialogue.h"

namespace {
inline void genNameUsingIndex(const String & prefix, int index, String &out_name)
{
	out_name= prefix + StringConverter::toString(static_cast<int>(index));
}
};

DIGIT_STRING_DIALOGUE::DIGIT_STRING_DIALOGUE(SceneManager *s)
{
	mSceneMgr = s;
	mMaterialName = "Examples/Digits";
	createObjs(mMaterialName);
}

DIGIT_STRING_DIALOGUE::DIGIT_STRING_DIALOGUE(SceneManager *s, const String &a_Material_Name)
{
	mMaterialName = a_Material_Name;
	mSceneMgr = s;
	createObjs(mMaterialName);
}

void DIGIT_STRING_DIALOGUE::createObjs(const String &a_Material_Name)
{
	int i, j;
	String name;
	for (j = 0; j < 5; j++) {
		for (i = 0; i < 10; i++) {
			int index = i + j*10;			
			genNameUsingIndex("fName", index, name);
			fr[index] = new FilledRectangle(name, mMaterialName, i);
			mSceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(fr[index]);
			fr[index]->setVisible(false);
		}
	}
	
	//fr[0]->setMaterialName("Examples/Digits");
	//fr[0]->setMaterialName("Examples/Flare");
	/*
	r[0] = new SelectionRectangle("rName0");
	mSceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(r[0]);
	r[0]->setVisible(true);
	*/
	setScore(123);
	
}

/*
	
*/
void DIGIT_STRING_DIALOGUE::setScore(int score, Real x, Real y, Real dx, Real dy)
{
	
	char msg[64];
	sprintf(msg, "%d", score);
	int len = strlen(msg);
	
	for (int j = 0; j < 50; j++) {
		fr[j]->setVisible(false);
	}
	if (len >=5 ) len = 5;
	for (int i = 0; i < len; i++, x += dx) {
		int index = i*10 + msg[i] -'0';
		fr[index]->setCorners(Vector2(x, y), Vector2(x+dx, y+dy));
		fr[index]->setVisible(true);
	}
}