/*
This is a game demo written by Wingo Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#include "bar2D.h"
#include "ogre_utility_3D2D.h"


BAR_2D::BAR_2D(SceneManager *sceneMgr, const String &materialName, const SceneNode *sceneNode, uint8 renderQueue)
{
	mSceneMgr = sceneMgr;
	mSceneNode = sceneNode;
	mMaterialName = materialName;
	mRenderQueue = renderQueue;
	mNode = NULL;
	mBarWidth = 0.25;
	mBarHeight = 0.025;
	mBar = NULL;
	mCur_value = 100;
	mMax_value = 100;
	//
	default_pos_x = 0.3;
	default_pos_y = 0.01;
	mPlacementMethod = BAR_2D_PLACEMENT_METHOD_FIXED;
	mCameraNode = NULL;
	mUseIdentity = true;
	init();
}

BAR_2D::BAR_2D(SceneManager *sceneMgr, const String &materialName, const SceneNode *sceneNode, const SceneNode *camNode, bool flg_use_identity, uint8 renderQueue)
{
	mSceneMgr = sceneMgr;
	mSceneNode = sceneNode;
	mNode = NULL;
	mCameraNode = camNode;
	mMaterialName = materialName;
	mRenderQueue = renderQueue;
	
	mBarWidth = 0.25;
	mBarHeight = 0.025;
	mBar = NULL;
	mCur_value = 100;
	mMax_value = 100;
	//
	mUseIdentity = flg_use_identity;
	default_pos_x = 0.3;
	default_pos_y = 0.01;
	mPlacementMethod = BAR_2D_PLACEMENT_METHOD_FIXED;
	if (mSceneNode) {
		mPlacementMethod = BAR_2D_PLACEMENT_METHOD_ATTACHED_OBJ;
	}
	
	init();
	
	
}

void BAR_2D::init()
{
	String name;
	generateGlobalObjName(name);
	//
	if (mBar == NULL) mBar = new FilledRectangle_Bar(name, mMaterialName, mRenderQueue, mUseIdentity); 
	mBar->setCorners(0, 0, mBarWidth, mBarHeight);
    mBar->setCastShadows(false);
	mBar->setVisible(true);
	//
	mNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	mNode->attachObject(mBar);

	mBar->setVisibilityFlags(0x01);
}

void BAR_2D::reset()
{
	mCur_value = mMax_value;
	update(0, 0);
}

void BAR_2D::setInfo(Real cur_value, Real max_value)
{
	mCur_value = cur_value;
	if (mCur_value < 0) mCur_value = 0;
	mMax_value = max_value;
}		

void BAR_2D::setSourceSceneNode(const SceneNode *sceneNode)
{
	mSceneNode = sceneNode;
}

/*
(x, y): upper left corner
dx, dy: width, height
*/
void BAR_2D::setBar(Real x, Real y, Real dx, Real dy)
{
	mBar->setInfo(mCur_value, mMax_value);

	mBar->setCorners(x, y, x + dx, y + dy);
}

void BAR_2D::setBarDimension(Real width, Real height)
{
	mBarWidth = width;
	mBarHeight = height;
}

void BAR_2D::setDefaultPos(Real x, Real y)
{
	default_pos_x = x;
	default_pos_y = y;
}

void BAR_2D::update(const Camera *camera, Real time_step, Real xoffset, Real yoffset)
{
	

	BAR_2D_PLACEMENT_METHOD method = mPlacementMethod;

	switch(mPlacementMethod) {
		case BAR_2D_PLACEMENT_METHOD_NONE:
		case BAR_2D_PLACEMENT_METHOD_FIXED:
			break;
		case BAR_2D_PLACEMENT_METHOD_HORIZONTAL_CENTER:
			break;
		case BAR_2D_PLACEMENT_METHOD_ATTACHED_OBJ:
			//if (mSceneNode == NULL) method = BAR_2D_PLACEMENT_METHOD_NONE;
			break;

	}
	Real xx, yy;
	switch(method) {
		case BAR_2D_PLACEMENT_METHOD_NONE:
		case BAR_2D_PLACEMENT_METHOD_FIXED:
			setBar(default_pos_x, default_pos_y, mBarWidth, mBarHeight);
			break;
		case BAR_2D_PLACEMENT_METHOD_HORIZONTAL_CENTER:
			xx = (1 - mBarWidth)*0.5;;
			yy = default_pos_y;
			setBar(xx, yy, mBarWidth, mBarHeight);
			break;
		case BAR_2D_PLACEMENT_METHOD_ATTACHED_OBJ:
			setBar(-mBarWidth*0.5, -mBarHeight*0.5, mBarWidth, mBarHeight);	
			mNode->setOrientation(camera->getOrientation());
			mNode->setPosition(mSceneNode->getPosition());
			break;
			Vector3 center_pos = mSceneNode->getPosition();
			Vector2 screen_pos;
			
			mapWorldPositionToViewportCoordinates(camera, center_pos, screen_pos);

			xx = screen_pos.x - mBarWidth*0.5 + xoffset;
			yy = screen_pos.y + yoffset;

			setBar(xx, yy, mBarWidth, mBarHeight);	
			
			break;
	}

}

void BAR_2D::setPlacementMethod(BAR_2D_PLACEMENT_METHOD m)
{
	mPlacementMethod = m;

}