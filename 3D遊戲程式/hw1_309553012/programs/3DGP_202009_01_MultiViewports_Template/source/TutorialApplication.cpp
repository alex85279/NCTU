////////////////////////////////////////
//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
// Date: 2020/09/25
////////////////////////////////////////
// Student Name: Chien-Chou Wong
// Student ID: 309553012
// Student Email: alexwong85279@gmail.com
//
////////////////////////////////////////
// You can delete or add some functions to do the assignment.
////////////////////////////////////////

#include "TutorialApplication.h"
#include "BasicTools.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace Ogre;

const float PI = 3.141592654;
double sc = 0;
BasicTutorial_00::BasicTutorial_00(void) {}

void BasicTutorial_00::chooseSceneManager()
{
	mSceneMgrArr[0] = mRoot
		->createSceneManager(ST_GENERIC, "primary");
	mSceneMgrArr[1] = mRoot
		->createSceneManager(ST_GENERIC, "secondary");

}

void BasicTutorial_00::createCamera_00(void)
{
	mSceneMgr = mSceneMgrArr[0];
	mCamera = mCameraArr[0] = mSceneMgr->createCamera("PlayerCam");
	mCamera->setPosition(Ogre::Vector3(120,300,600));
	mCamera->lookAt(Ogre::Vector3(120,0,0));
	mCamera->setNearClipDistance(5);
	mCameraManArr[0] = new OgreBites::SdkCameraMan(mCamera);   
}

void BasicTutorial_00::createCamera_01(void)
{
	// add your own stuff
	mSceneMgr = mSceneMgrArr[1];
	mCamera = mCameraArr[1] = mSceneMgr->createCamera("MapCam");
	mCamera->setPosition(Ogre::Vector3(0,350,0.001));
	mCamera->lookAt(Ogre::Vector3(0,0,0));
	mCamera->setNearClipDistance(5);
	mCameraManArr[1] = new OgreBites::SdkCameraMan(mCamera);

}



void BasicTutorial_00::createViewport_00(void)
{
	mCamera = mSceneMgrArr[0]->getCamera("PlayerCam");
	Viewport* vp = mWindow->addViewport(mCamera,0,0,0,1,1);
	vp->setBackgroundColour(ColourValue(0,0,1));
	mCamera->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
	mViewportArr[0] = vp;
}

void BasicTutorial_00::createViewport_01(void)
{
    // add your own stuff
	mCamera = mSceneMgrArr[1]->getCamera("MapCam");
	Viewport* vp = mWindow->addViewport(mCamera, 1, 0, 0, 0.25f,0.25f);
	vp->setBackgroundColour(ColourValue(0,1,0));
	vp->setOverlaysEnabled(false);
	mCamera->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
	mViewportArr[1] = vp;
}

void BasicTutorial_00::createScene_00(void) 
{
	mSceneMgr = mSceneMgrArr[0];
	//ambient light
	mSceneMgr->setAmbientLight(ColourValue(0.01,0.01,0.01));
	
	//shadow
	mSceneMgr->setShadowTechnique(SHADOWTYPE_STENCIL_ADDITIVE);
	
	//plane
	Plane plane(Vector3::UNIT_Y,0);
	MeshManager::getSingleton().createPlane(
		"ground",
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
		plane,
		1500,1500,
		20,20,
		true,
		1,
		5,5,
		Vector3::UNIT_Z
		
		);
	Entity *plane_ent = mSceneMgr->createEntity("Ground","ground");
	plane_ent->setCastShadows(false);
	mSceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(plane_ent);
	// add penguin_large
	Entity *penguin_lg_ent = mSceneMgr->createEntity("Penguin1","penguin.mesh");
	penguin_lg_ent->setCastShadows(true);
	SceneNode *node1 = mSceneMgr->getRootSceneNode()->createChildSceneNode("PenguinNode1", Vector3(0,50,0));
	node1->attachObject(penguin_lg_ent);
	node1->scale(2,3,2);

	// add penguin_small
	Entity *penguin_sl_ent = mSceneMgr->createEntity("Penguin2","penguin.mesh");
	SceneNode *node2 = mSceneMgr->getRootSceneNode()->createChildSceneNode("PenguinNode2", Vector3(200,50,0));
	node2->attachObject(penguin_sl_ent);
	
	//rotate penguin
	Vector3 pen1_pos = node1->getPosition();
	Vector3 pen2_pos = node2->getPosition();
	Vector3 cam_pos = mCameraMan->getCamera()->getPosition();
	node1->lookAt(cam_pos,Node::TS_PARENT, Vector3::UNIT_Z);
	node2->lookAt(pen1_pos,Node::TS_PARENT, Vector3::UNIT_Z);
	
	//circle
	int cubes = 144;
	int L = 255;
	for(int i = 0; i<cubes;i++){
		String name;
		genNameUsingIndex("c", i , name);
		Entity *ent = mSceneMgr->createEntity(name,"cube.mesh");
		ent->setMaterialName("Examples/SphereMappedRustSteel");
		AxisAlignedBox bb = ent->getBoundingBox();
		int cubeSize = bb.getMaximum().x - bb.getMinimum().x;
		int x,y,z;
		x = 0, y = 0, z = -125;

		SceneNode *snode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
		snode->attachObject(ent);

		double fx = i/(double)(cubes-1);
		double h = (1 + sin(fx * PI * 4)) * 50;
		double wave = (1 + 2*sin(fx * PI * 8)) * 50;
		float r = 100 + wave;
		float x1 = r * cos(fx * PI * 2);
		float z1 = r * sin(fx * PI * 2);
		float unitF = 1.0/cubeSize/cubes*L*0.8;
		snode->scale(unitF, h/cubeSize, unitF);
		snode->setPosition(x1,50,z1);

	}
	int cube2 = 20;
	//row
	for(int i = 0; i<cube2;i++){
		String name;
		genNameUsingIndex("rc", i , name);
		Entity *ent = mSceneMgr->createEntity(name,"cube.mesh");
		ent->setMaterialName("Examples/Chrome");
		AxisAlignedBox bb = ent->getBoundingBox();
		int cubeSize = bb.getMaximum().x - bb.getMinimum().x;
		int x,y,z;
		x = 0, y = 0, z = -125;

		SceneNode *snode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
		snode->attachObject(ent);
		double fx = 2*i/(double) (cube2-1); 
		float cx = fx*L - L/2.0;
		float h = (1+cos(fx*3.1415*2.0))*20; 
		Real unitF = 1.0/cubeSize/cube2*L*0.8;
		snode->scale(unitF, h/cubeSize, unitF);
		snode->setPosition(cx, 20, z);
	
	
	}

	//light
	Light *light = mSceneMgr->createLight("Light1");
	light->setType(Light::LT_POINT); 
	light->setPosition(Vector3(150, 250, 100)); 
	light->setDiffuseColour( ColourValue(1,0,0) );		
	light->setSpecularColour( ColourValue(1,0,0) );	

	SceneNode *lnode1 = mSceneMgr->getRootSceneNode()->createChildSceneNode("lightNode1", Vector3(50,200,0));
	lnode1->attachObject(light);


	Light *light2 = mSceneMgr->createLight("Light2"); //error
	light2->setType(Light::LT_POINT);  //error
	light2->setPosition(Vector3(-150, 300, 250));  //error
	light2->setDiffuseColour(ColourValue(0,0,1));		
	light2->setSpecularColour(ColourValue(0,0,1));
	SceneNode *lnode2 = mSceneMgr->getRootSceneNode()->createChildSceneNode("lightNode2", Vector3(0,200,50));
	lnode2->attachObject(light2);
    // add your own stuff
}

void BasicTutorial_00::createScene_01(void) 
{
    // add your own stuff
	
	mSceneMgr = mSceneMgrArr[1];
	//ambient light
	mSceneMgr->setAmbientLight(ColourValue(0.1,0.1,0.1));
	
	//shadow
	mSceneMgr->setShadowTechnique(SHADOWTYPE_STENCIL_ADDITIVE);
	
	//plane
	Plane plane(Vector3::UNIT_Y,0);
	MeshManager::getSingleton().createPlane(
		"ground2",
		ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
		plane,
		1500,1500,
		20,20,
		true,
		1,
		5,5,
		Vector3::UNIT_Z
		
		);
	Entity *plane_ent = mSceneMgr->createEntity("Ground","ground2");
	plane_ent->setCastShadows(false);
	mSceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(plane_ent);

	Light *light3 = mSceneMgr->createLight("Light3");
	light3->setType(Light::LT_POINT);
	light3->setPosition(Vector3(100, 150, 250));
	light3->setDiffuseColour(ColourValue(0,0,1));		
	light3->setSpecularColour(ColourValue(0,0,1));
	SceneNode *lnode3 = mSceneMgr->getRootSceneNode()->createChildSceneNode("lightNode3", Vector3(0,200,50));
	lnode3->attachObject(light3);



	Entity *ent = mSceneMgr->createEntity("test","cube.mesh");
	ent->setMaterialName("Examples/green");
	SceneNode *snode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
	snode->attachObject(ent);
	snode->setPosition(0, 0.5, 0);
	snode->scale(0.5, 0.5, 0.5);
}

void BasicTutorial_00::createViewports(void)
{
    //Do not modify
	createViewport_00();
	createViewport_01();
}

void BasicTutorial_00::createCamera(void) {
    //Do not modify
	createCamera_00();
	createCamera_01();
	mCameraMan = mCameraManArr[0];
	//mCameraMan = mCameraManArr[1];
}

void BasicTutorial_00::createScene( void ) {
    //Do not modify
	createScene_00();
	createScene_01();
	//mSceneMgr = mSceneMgrArr[0]; // active SceneManager
	mSceneMgr = mSceneMgrArr[1]; // active SceneManager
    //
    mCamera = mCameraArr[0];
    //mCamera = mCameraArr[1];
	mAngle = 0;
	mAngularSpeed = 0;
	reverse = 1;
	toggle = false;
	place_idx = 3;
}

//
// What is stored in the file for arg.key?
// ASCII code? If no, what is it?
//
// To find out the answer:
// Go to see the definition of KeyEvent
//
bool BasicTutorial_00::keyPressed( const OIS::KeyEvent &arg )
{
    bool flg = true;
    stringstream ss;
    ss << arg.key;
    String msg;
    ss >> msg;
    msg += ":*** keyPressed ***\n";
    Ogre::LogManager::getSingletonPtr()->logMessage( msg );

    
    if (arg.key == OIS::KC_C ) {
        
        //How to clear ss?
        ss.str("");
        ss.clear();
        
        //stringstream ss; // Second way

        // Third way?
        //=============

        // How to get camerea position?
        //-----------------------------
        //This is incorrect.
        //Vector3 pos = mCamera->getPosition();
        //-----------------------------
        Vector3 pos = mCameraMan->getCamera()->getPosition(); //Correct
        ss << std::fixed << std::setprecision(2) 
            << "CameraPosition:" 
            << pos.x << "\t" 
            << pos.y << "\t" 
            << pos.z << "\n";
        Ogre::LogManager::getSingletonPtr()
            ->logMessage( ss.str() );
        //
        ss.str("");
        ss.clear();
        Vector3 dir = mCameraMan->getCamera()->getDirection();
        ss << std::fixed << std::setprecision(2) 
            << "CameraDirection:" 
            << dir.x << "\t" 
            << dir.y << "\t" 
            << dir.z << "\n";
        Ogre::LogManager::getSingletonPtr()
            ->logMessage( ss.str() );
        //
    }

    if (arg.key == OIS::KC_1 ) {
        mCameraMan->getCamera()
            ->setPosition(Vector3(98.14,	450.69,	964.20));
        mCameraMan->getCamera()
            ->setDirection(Vector3(-0.01,	-0.30,	-0.95));

        //98.14	450.69	964.20
        //-0.01	-0.30	-0.95
    }

    if (arg.key == OIS::KC_2 ) {
        // add your own stuff
		mCameraMan->getCamera()
            ->setPosition(Vector3(-1463.00,	606.45,	-513.24));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.88,	-0.48,	0.10));
        //-1463.00	606.45	-513.24
        //0.88	-0.47	0.10
    }

    if (arg.key == OIS::KC_3 ) {
        // add your own stuff
		mCameraMan->getCamera()
            ->setPosition(Vector3(-1356.16,	634.32,	-964.51));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.71, -0.44,	0.55));
        //-1356.16	634.32	-964.51
        //0.71	-0.44	0.55
    }

    if (arg.key == OIS::KC_4 ) {
         // add your own stuff
		mCameraMan->getCamera()
            ->setPosition(Vector3(40.39, 155.23, 251.20));
        mCameraMan->getCamera()
            ->setDirection(Vector3(-0.02, -0.41, -0.91));
        //40.39	155.23	251.20
        //-0.02	-0.41	-0.91
    }

    if (arg.key == OIS::KC_5 ) {
        // add your own stuff
		mCameraMan->getCamera()
            ->setPosition(Vector3(19.94, 822.63, 30.79));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.00, -0.99, -0.11));
        //19.94	822.63	30.79
        //0.00	-0.99	-0.11
    }

    if (arg.key == OIS::KC_M ) {

        Camera *c_ptr = mCameraArr[0];
		Camera *c_ptr1 = mCameraArr[1];
        mWindow->removeViewport(mViewportArr[0]->getZOrder());
	    mWindow->removeViewport(mViewportArr[1]->getZOrder());

	    Viewport* vp = mWindow->addViewport(c_ptr,1,0,0,0.25,0.25);
	    vp->setBackgroundColour(ColourValue(0,0,1));
		c_ptr->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
		mViewportArr[0] = vp;

		vp = mWindow->addViewport(c_ptr1, 0, 0, 0, 1, 1);
		vp->setBackgroundColour(ColourValue(0,1,0));
		vp->setOverlaysEnabled(false);
		c_ptr1->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
		mViewportArr[1] = vp;

		place_idx = 0;
    }

    if (arg.key == OIS::KC_N ) {
        // add your own stuff
		Camera *c_ptr = mCameraArr[0];
		Camera *c_ptr1 = mCameraArr[1];
        mWindow->removeViewport(mViewportArr[0]->getZOrder());
	    mWindow->removeViewport(mViewportArr[1]->getZOrder());

	    Viewport* vp = mWindow->addViewport(c_ptr,0,0,0,1,1);
	    vp->setBackgroundColour(ColourValue(0,1,0));
		c_ptr->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
		mViewportArr[0] = vp;

		vp = mWindow->addViewport(c_ptr1, 1, 0.75f, 0, 0.25f, 0.25f);
		vp->setBackgroundColour(ColourValue(0,0,1));
		vp->setOverlaysEnabled(false);
		c_ptr1->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
		mViewportArr[1] = vp;
		place_idx = 1;
    }
	if (arg.key == OIS::KC_B ) {
        // add your own stuff
		
		int order0 = mViewportArr[0]->getZOrder();
		int order1 = mViewportArr[1]->getZOrder();
		int rm_window = (order0 == 1)? 0 : 1;
		Camera *c_ptr = mCameraArr[rm_window];
		mWindow->removeViewport(mViewportArr[rm_window]->getZOrder());
		place_idx ++;
		place_idx = place_idx % 4;
		Viewport *vp;
		if(place_idx == 0) vp = mWindow->addViewport(c_ptr, 1, 0, 0, 0.25f, 0.25f);
		else if(place_idx == 1) vp = mWindow->addViewport(c_ptr, 1, 0.75f, 0, 0.25f, 0.25f);
		else if(place_idx == 2) vp = mWindow->addViewport(c_ptr, 1, 0.75f, 0.75f, 0.25f, 0.25f);
		else vp = mWindow->addViewport(c_ptr, 1, 0, 0.75f, 0.25f, 0.25f);
		vp->setBackgroundColour(ColourValue(0,1,0));
		if(rm_window == 1)vp->setOverlaysEnabled(false);
		c_ptr->setAspectRatio(Real(vp->getActualWidth())/Real(vp->getActualHeight()));
		mViewportArr[rm_window] = vp;

		
    }
	if (arg.key == OIS::KC_P ){
		toggle = !toggle;
	
	}

    // Do not delete this line
    BaseApplication::keyPressed(arg);

    return flg;
}

//
// What is stored in the file for arg.key?
// ASCII code? If no, what is it?
// 
// To find out the answer:
// Go to see the definition of KeyEvent
//
bool BasicTutorial_00::keyReleased( const OIS::KeyEvent &arg )
{
    bool flg = true;
    stringstream ss;
    ss << arg.key;
    String msg;
    ss >> msg;
    msg += ":*** keyReleased ***\n";
    
    Ogre::LogManager::getSingletonPtr()->logMessage( msg );

    BaseApplication::keyReleased(arg);

    return flg;
}

bool BasicTutorial_00::frameStarted(const Ogre::FrameEvent& evt)
{
	bool flg = Ogre::FrameListener::frameStarted(evt);
    //
    // add your own stuff
    //
	
	if(toggle == true){
		mSceneMgr = mSceneMgrArr[0];
		SceneNode *node1 = mSceneMgr->getSceneNode("PenguinNode1");
		SceneNode *node2 = mSceneMgr->getSceneNode("PenguinNode2");
		Vector3 pen1_pos = node1->getPosition();
		Vector3 pen2_pos = node2->getPosition();
		node1->lookAt(pen2_pos,Node::TS_WORLD, Vector3::UNIT_Z);

		/*
		double fx = i/(double)(cubes-1);
		double h = (1 + sin(fx * PI * 4)) * 50;
		float r = 100;
		float x1 = r * cos(fx * PI * 2);
		float z1 = r * sin(fx * PI * 2);
		float unitF = 1.0/cubeSize/cubes*L*0.8;
		snode->scale(unitF, h/cubeSize, unitF);
		snode->setPosition(x1,50,z1);
		*/
		float r = pen1_pos.distance(pen2_pos);
	
		pen2_pos.x = r * cos(mAngle * reverse);
		pen2_pos.z = r * sin(mAngle * reverse);
		node2->setPosition(pen2_pos);
	
		mAngularSpeed += 0.1 * evt.timeSinceLastFrame;
		mAngle += mAngularSpeed * evt.timeSinceLastFrame;

		if(mAngle >= 2 * PI){
			mAngle = 0;
			mAngularSpeed = 0;
			if(reverse == 1) reverse = -1;
			else reverse = 1;
		}
	}
	//pen2_pos.x = r * cos
	

	//node2->yaw(Degree(-1));
		
    return flg;
}
int main(int argc, char *argv[]) {
	BasicTutorial_00 app;
	app.go();  
	return 0;
}

////////////////////////////////////////
// DO NOT DELETE THIS LINE: 2018/09/20. 3D Game Programming
////////////////////////////////////////