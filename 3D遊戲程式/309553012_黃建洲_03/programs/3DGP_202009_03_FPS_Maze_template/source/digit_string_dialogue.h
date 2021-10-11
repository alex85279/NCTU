/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#ifndef __DIGIT_STRING_DIALOGUE_BOX__
#define __DIGIT_STRING_DIALOGUE_BOX__

#include "Ogre.h"
#include "OgreStringConverter.h"
#include "OgreException.h"

using namespace Ogre;

/*
	How to use DIGIT_STRING_DIALOGUE?
	(1) Create it with a scene manager
	(2) call setScore
*/
class SelectionRectangle : public ManualObject
{
public:
    SelectionRectangle(const String &name)
        : ManualObject(name)
    {
        setUseIdentityProjection(true);
        setUseIdentityView(true);
        setRenderQueueGroup(RENDER_QUEUE_OVERLAY);
        setQueryFlags(0);
    }
 
    /**
     * Sets the corners of the SelectionRectangle.  Every parameter should be in the
     * range [0, 1] representing a percentage of the screen the SelectionRectangle
     * should take up.
     */
    void setCorners(float left, float top, float right, float bottom)
    {
        left = left * 2 - 1;
        right = right * 2 - 1;
        top = 1 - top * 2;
        bottom = 1 - bottom * 2;

        clear();
        begin("", RenderOperation::OT_LINE_STRIP);
            position(left, top, -1);
            position(right, top, -1);
            position(right, bottom, -1);
            position(left, bottom, -1);
            position(left, top, -1);
        end();

        AxisAlignedBox box;
        box.setInfinite();
        setBoundingBox(box);
		
    }

    void setCorners(const Vector2 &topLeft, const Vector2 &bottomRight)
    {
        setCorners(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

};

class FilledRectangle : public ManualObject
{
	void computeTextCoordinate(int digit, 
		Vector2 &bottomLeft, Vector2 &topRight) {
		Real x, y;
		bottomLeft.x = 0;
		bottomLeft.y = 1;
		topRight.x = 1;
		topRight.y = 0;

		Real dx = 1/10.0;
		bottomLeft.x = dx*digit;
		topRight.x = bottomLeft.x + dx;
		if (1) {
			//do correction
			topRight.x = bottomLeft.x + dx*0.9;
		}
		if (digit == 0) {
			//do correction
			bottomLeft.x = dx*digit +dx*0.05;
		}
		if (digit == 1) {
			//do correction
			topRight.x = topRight.x +dx*0.1;
		}
		if (digit == 2) {
			//do correction
			bottomLeft.x = bottomLeft.x +dx*0.05;
		}
		if (digit == 3) {
			//do correction
			bottomLeft.x = bottomLeft.x -dx*0.1;
		}
	}
public:
    FilledRectangle(const String &name, int digit)
        : ManualObject(name)
    {
		mDigit = digit;
		setVisibilityFlags(0x01);
        setUseIdentityProjection(true);
        setUseIdentityView(true);
        setRenderQueueGroup(RENDER_QUEUE_OVERLAY);
        setQueryFlags(0);
		mMaterial_Name = "Examples/Digits";
    }
	FilledRectangle(const String &name, const String &a_Material_Name, int digit)
        : ManualObject(name)
    {
		mDigit = digit;
		setVisibilityFlags(0x01);
        setUseIdentityProjection(true);
        setUseIdentityView(true);
        setRenderQueueGroup(RENDER_QUEUE_OVERLAY);
        setQueryFlags(0);
		mMaterial_Name = a_Material_Name;
    }
 
    /**
     * Sets the corners of the SelectionRectangle.  Every parameter should be in the
     * range [0, 1] representing a percentage of the screen the SelectionRectangle
     * should take up.
     */
    void setCorners(float left, float top, float right, float bottom)
    {
        left = left * 2 - 1;
        right = right * 2 - 1;
        top = 1 - top * 2;
        bottom = 1 - bottom * 2;

		Vector2 bottomLeft, topRight;
		computeTextCoordinate(mDigit, bottomLeft, topRight);
        clear();
		begin(mMaterial_Name, RenderOperation::OT_TRIANGLE_FAN);
		/*
            position(left, top, -1);
            position(right, top, -1);
            position(right, bottom, -1);
            position(left, bottom, -1);
			*/
		position(left, bottom, -1);
		textureCoord(bottomLeft.x, bottomLeft.y);
		position(right, bottom, -1);
		textureCoord(topRight.x, bottomLeft.y);
		position(right, top, -1);
		textureCoord(topRight.x, topRight.y);
		position(left, top, -1);
		textureCoord(bottomLeft.x, topRight.y);     
        end();

        AxisAlignedBox box;
        box.setInfinite();
        setBoundingBox(box);
		
    }

    void setCorners(const Vector2 &topLeft, const Vector2 &bottomRight)
    {
        setCorners(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }
private:
	int mDigit;
	String mMaterial_Name;

};

class DIGIT_STRING_DIALOGUE {
private:
	SceneManager *mSceneMgr;
	String mMaterialName;
	SelectionRectangle *r[3];
	FilledRectangle *fr[50];
	void createObjs(const String &a_Material_Name);

public:
	DIGIT_STRING_DIALOGUE(SceneManager *s);
	DIGIT_STRING_DIALOGUE(SceneManager *s, const String &a_Material_Name);
	void setScore(int score, Real x = 0, Real y = 0, Real dx = 0.025, Real dy = 0.05);
};

#endif