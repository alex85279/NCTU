/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#ifndef __BAR_2D_H__
#define __BAR_2D_H__



#include "Ogre.h"
#include "OgreStringConverter.h"
#include "OgreException.h"
#include "game_obj_name_factory.h"

using namespace Ogre;
/*
*/
class FilledRectangle_Bar : public ManualObject
{
	void computeTextCoordinate(int digit, 
		Vector2 &bottomLeft, Vector2 &topRight) {
		bottomLeft.x = 0;
		bottomLeft.y = 1;
		topRight.x = 1;
		topRight.y = 0;


	}
public:
    FilledRectangle_Bar(const String &name, const String &materialName, uint8 renderQueue = RENDER_QUEUE_8, bool use_identity = true)
        : ManualObject(name)
    {
		mMaterialName = materialName;
		mRenderQueue = renderQueue;
		
		//if (use_identity) {
			setUseIdentityProjection(use_identity);
			setUseIdentityView(use_identity);
			//
			setRenderQueueGroup(mRenderQueue);
		//} else {
		//	setRenderQueueGroup(RENDER_QUEUE_MAIN);
		//}
		
        setQueryFlags(0);
		mFlgSplit = true;
		setCorners(0, 0, 1, 1);
		mCur_value = mMax_value = 100;
    }
 
    /**
     * Sets the corners of the SelectionRectangle.  Every parameter should be in the
     * range [0, 1] representing a percentage of the screen the SelectionRectangle
     * should take up.
     */
    void _setCorners_split(float left, float top, float right, float bottom)
    {
        left = left * 2 - 1;
        right = right * 2 - 1;
        top = 1 - top * 2;
        bottom = 1 - bottom * 2;
		//
		Real f; 
		if (mMax_value == 0) {
			f = 0;
		} else {
			f = mCur_value / mMax_value;
		}
		Real mid;
		mid = left + f*(right - left);
		//
        clear();
		begin(mMaterialName, RenderOperation::OT_TRIANGLE_LIST);

		
		position(mid, bottom, -1);  // 0
		textureCoord(0.6,0);  
		position(right, bottom, -1); // 1
		textureCoord(0.9, 0);
		position(right, top, -1); // 2
		textureCoord(0.9, 1);

		//
		position(mid, bottom, -1);  // 3
		textureCoord(0.6,0);  
		position(right, top, -1); // 4
		textureCoord(0.9, 1);
		position(mid, top, -1); // 5
		textureCoord(0.6, 1);
		//
		position(mid, bottom, -1);  // 6
		textureCoord(0.4,0);  
		position(mid, top, -1); // 7
		textureCoord(0.4, 0);
		position(left, top, -1); // 8
		textureCoord(0.1, 1);
		//
		position(mid, bottom, -1);  // 9
		textureCoord(0.4, 0);  
		position(left, top, -1); // 10
		textureCoord(0.1, 1);
		position(left, bottom, -1); // 11
		textureCoord(0.1, 0);

		//
        end();

        AxisAlignedBox box;
        box.setInfinite();
        setBoundingBox(box);
		
    }

	   void _setCorners_nonsplit(float left, float top, float right, float bottom)
    {
        left = left * 2 - 1;
        right = right * 2 - 1;
        top = 1 - top * 2;
        bottom = 1 - bottom * 2;
		//
		//
        clear();
		begin(mMaterialName, RenderOperation::OT_TRIANGLE_LIST);

		position(left, bottom, -1);  // 0
		textureCoord(0.0,0);  
		position(right, bottom, -1); // 1
		textureCoord(1.0, 0);
		position(right, top, -1); // 2
		textureCoord(1.0, 1);
		//
		position(left, bottom, -1);  // 3
		textureCoord(0.0,0);  
		position(right, top, -1); // 4
		textureCoord(1.0, 1);
		position(left, top, -1); // 5
		textureCoord(0.0, 1);

		//
        end();

        AxisAlignedBox box;
        box.setInfinite();
        setBoundingBox(box);
		
    }

	void setCorners(float left, float top, float right, float bottom) {
		if (mFlgSplit) {
			_setCorners_split(left, top, right, bottom);
		} else {
			_setCorners_nonsplit(left, top, right, bottom);
		}

	}
    void setCorners(const Vector2 &topLeft, const Vector2 &bottomRight)
    {
        setCorners(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

	void setInfo(Real cur_value, Real max_value) {
		mCur_value = cur_value;
		mMax_value = max_value;
	}
	void setRenderQueue(uint8 q) {
		mRenderQueue = q;
		setRenderQueueGroup(mRenderQueue);
	}
	void setSplit2Parts(bool flg) {
		mFlgSplit = flg;
	}

private:
	bool mFlgSplit;
	Real mCur_value, mMax_value;
	int mDigit;
	String mMaterialName;
	uint8 mRenderQueue;

};


typedef enum {
	BAR_2D_PLACEMENT_METHOD_NONE,
	BAR_2D_PLACEMENT_METHOD_HORIZONTAL_CENTER,
	BAR_2D_PLACEMENT_METHOD_ATTACHED_OBJ,
	BAR_2D_PLACEMENT_METHOD_FIXED
} BAR_2D_PLACEMENT_METHOD;

class BAR_2D : protected GAME_OBJ_NAME_FACTORY{
protected:

private:
	uint8 mRenderQueue;
	FilledRectangle_Bar *mBar;
	SceneManager *mSceneMgr;
	SceneNode *mNode;
	Real mCur_value;
	Real mMax_value;
	Real default_pos_x, default_pos_y;
	Real mBarWidth, mBarHeight;
	const SceneNode *mSceneNode;
	String mMaterialName;
	String mPrefixName;
	const SceneNode *mCameraNode;
	bool mUseIdentity;
	//
	BAR_2D_PLACEMENT_METHOD mPlacementMethod;
private:
	void init();
	void setBar(Real x, Real y, Real dx, Real dy); // corner coord + width and height
public:
	BAR_2D(SceneManager *sceneMgr, const String &materialName, const SceneNode *sceneNode, uint8 renderQueue = RENDER_QUEUE_8);
	BAR_2D(SceneManager *sceneMgr, const String &materialName, const SceneNode *sceneNode, const SceneNode *camNode, bool flg_use_identity = true, uint8 renderQueue = RENDER_QUEUE_8);
	void setSplit2Parts(bool flg) {
		mBar->setSplit2Parts(flg);
	}
	void update(const Camera *camera, Real time_step, Real xoffset = 0, Real yoffset = 0);
	void setBarDimension(Real width, Real height);
	void reset();
	void setDefaultPos(Real x, Real y);
	void setSourceSceneNode(const SceneNode *sceneNode);
	void setInfo(Real cur_value, Real max_value);
	void getInfo(Real &cur_value, Real &max_value) {
		cur_value = mCur_value;
		max_value = mMax_value;
	}
	void setVisible(bool flg) {
		if (mBar) {
			mBar->setVisible(flg);
		}
	}
	void setVisibilityFlags(unsigned int flg) { if (mBar) mBar->setVisibilityFlags(flg); }
	void setPlacementMethod(BAR_2D_PLACEMENT_METHOD m);
};

#endif