//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
// Date: 2019/10/28
//
#include "TutorialApplication.h"
#include "BasicTools.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace Ogre;

const float PI = 3.141592654;

#define mMoveDirection_NONE 0
#define mMoveDirection_DOWN (1 << 0)
#define mMoveDirection_UP   (1 << 1)
#define mMoveDirection_LEFT (1 << 2)
#define mMoveDirection_RIGHT (1<<3)


BasicTutorial_00::BasicTutorial_00(void)
    : 
mMoveDirection(mMoveDirection_NONE)
{
	index = 0;
}

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
	mCameraManArr[0] = new OgreBites::SdkCameraMan(mCamera);   // create a default camera controller
}

void BasicTutorial_00::createCamera_01(void)
{
	mSceneMgr = mSceneMgrArr[1];
	mCamera = mCameraArr[1] = mSceneMgr->createCamera("PlayerCam");
	mCamera->setPosition(Ogre::Vector3(0,350,0.0));
	mCamera->lookAt(Ogre::Vector3(0.00001,0,0));
	mCamera->setNearClipDistance(5);
	mCameraManArr[1] = new OgreBites::SdkCameraMan(mCamera);   // create a default camera controller

}



void BasicTutorial_00::createViewport_00(void)
{
	mCamera = mCameraArr[0];
	Ogre::Viewport* vp = mWindow->addViewport(mCamera);
	vp->setBackgroundColour(Ogre::ColourValue(0,0.0,1.0));
	mCamera->setAspectRatio(
		Ogre::Real(vp->getActualWidth()) / Ogre::Real(vp->getActualHeight()));

    mViewportArr[0] = vp;
}

void BasicTutorial_00::createViewport_01(void)
{
}

void BasicTutorial_00::resolveCollision(
    SceneNode *nodeA, SceneNode *nodeB,
    float rA, float rB, float wA, float wB)
{
    Vector3 posA = nodeA->getPosition();
    Vector3 posB = nodeB->getPosition();
    float R = rA + rB;
    ///////////////////////
    // add your own stuff
    ///////////////////////
}

void BasicTutorial_00::resolveCollisionLargeSphere()
{
    float smallR = 15; // small sphere radius

    float largeR = 1.0/0.15*smallR; // large sphere radius

	float radius = smallR + largeR;
	for(int i = 0; i< mNumSpheres; i++){
		Vector3 distance = mLargeSphereSceneNode->getPosition() - mSceneNode[i]->getPosition();
		if(distance.length() > radius) continue;
		Vector3 fix_dis = distance * abs(radius - distance.length())/distance.length();
		fix_dis.y = 0.0;
		//std::cout<<"yes"<<std::endl;
		mSceneNode[i]->translate(-fix_dis * 1 * speed * 0.1f);
		
	}
	Vector3 pos = mLargeSphereSceneNode->getPosition();
	Vector3 pos2,pos3,pos4;
	pos2 = mLargeSphereSceneNode->getPosition();
	pos3 = mLargeSphereSceneNode->getPosition();
	pos4 = mLargeSphereSceneNode->getPosition();
	if(pos.x > 600 - 125){
		mLargeSphereSceneNode->translate(Vector3(-1,0,0) * speed);
	}
	if(pos2.x < -600 + 125){
		mLargeSphereSceneNode->translate(Vector3(1,0,0) * speed);
	}
	if(pos3.z > 600 - 125){
		mLargeSphereSceneNode->translate(Vector3(0,0,-1) * speed);
	}
	if(pos4.z < -600 + 125){
		mLargeSphereSceneNode->translate(Vector3(0,0,1) * speed);
	}
	
}

// perform collision handling for all pairs
void BasicTutorial_00::resolveCollisionSmallSpheres()
{
    float ri = 15; // sphere radius
    float rj = 15; // sphere radius
	float radius = ri + rj;
    for (int i = 0; i < mNumSpheres; ++i)
	{
		for (int j = i+1; j < mNumSpheres; ++j) {
			Vector3 distance = mSceneNode[i]->getPosition() - mSceneNode[j]->getPosition();
			if(distance.length() > radius) continue;
			Vector3 fix_dis = distance * abs(radius - distance.length())/distance.length();
			fix_dis.y = 0.0;
			mSceneNode[i]->translate(fix_dis * 0.5f * speed * 0.1f);
			mSceneNode[j]->translate(-fix_dis * 0.5f * speed * 0.1f);
            ///////////////////////
            // add your own stuff
            ///////////////////////

        }
    }
}

void BasicTutorial_00::resolveCollision()
{
    int num = 10;
    for (int i = 0; i < num;++i) {
        resolveCollisionSmallSpheres();
        resolveCollisionLargeSphere();
    }
}

// reset positions of all small spheres
void BasicTutorial_00::reset()
{
    for (int i = 0; i < mNumSpheres; ++i ) {
        ///////////////////////
        // add your own stuff
        ///////////////////////
    }
}

// create all spheres
// "Examples/red"
// "Examples/green"
// "Examples/blue"
// "Examples/yellow"
void BasicTutorial_00::createSpace()
{
    String name_en;
    String name_sn;
	
    mNumSpheres = 500;
    for (int i = 0; i < mNumSpheres; ++i ) {

	    genNameUsingIndex("R1", index, name_en);
		genNameUsingIndex("S1", index, name_sn);
		mEntity[index] = mSceneMgr
			->createEntity( name_en, "sphere.mesh" );
		int color = rand()%3;
		if(color == 0) mEntity[index]->setMaterialName("Examples/red");
		else if(color == 1) mEntity[index]->setMaterialName("Examples/green");
		else mEntity[index]->setMaterialName("Examples/blue");

		float rand_x = rand()%800 - 400;
		float rand_z = rand()%800 - 400;
		mSceneNode[index] = mSceneMgr->getRootSceneNode()->createChildSceneNode(name_sn, Vector3(rand_x, 0, rand_z));
		mSceneNode[index]->attachObject(mEntity[index]);
        mSceneNode[index]->scale(0.15,0.15,0.15);
		
		///////////////////////
        // add your own stuff
        ///////////////////////
		index++;
    /*switch(rand()%3) {
    case 0:
        mEntity[index]->setMaterialName("Examples/red");
        break;
    case 1:
        break;
    case 2:
        break;
    }*/

    //scale the small spheres
    //mSceneNode[index]->scale(0.15, 0.15, 0.15);
    }

	//barrels
	int offset = mNumSpheres;
    int mNumObstacles = 20;
    for (int i = 0; i < mNumObstacles; ++i ) {
        genNameUsingIndex("R1", index, name_en);
        genNameUsingIndex("S1", index, name_sn);
        mEntity[index] = mSceneMgr
            ->createEntity( name_en, "Barrel.mesh" );
		float pos_x = -600 + 60 * i;
		float pos_z = -600;

        mSceneNode[index] = mSceneMgr
            ->getRootSceneNode()
            ->createChildSceneNode( 
            name_sn, Vector3(pos_x, 0, pos_z)); 

        mSceneNode[index]->scale(10.0, 10.0	, 10.0);
        mSceneNode[index]->attachObject(mEntity[index]);
		index++;
    }
	for (int i = 0; i < mNumObstacles; ++i ) {
        genNameUsingIndex("R1", index, name_en);
        genNameUsingIndex("S1", index, name_sn);
        mEntity[index] = mSceneMgr
            ->createEntity( name_en, "Barrel.mesh" );
		float pos_x = -600 + 60 * i;
		float pos_z = 600;

        mSceneNode[index] = mSceneMgr
            ->getRootSceneNode()
            ->createChildSceneNode( 
            name_sn, Vector3(pos_x, 0, pos_z)); 

        mSceneNode[index]->scale(10.0, 10.0	, 10.0);
        mSceneNode[index]->attachObject(mEntity[index]);
		index++;
    }
	for (int i = 0; i < mNumObstacles; ++i ) {
        genNameUsingIndex("R1", index, name_en);
        genNameUsingIndex("S1", index, name_sn);
        mEntity[index] = mSceneMgr
            ->createEntity( name_en, "Barrel.mesh" );
		float pos_x = -600;
		float pos_z = -600 + 60 * i;

        mSceneNode[index] = mSceneMgr
            ->getRootSceneNode()
            ->createChildSceneNode( 
            name_sn, Vector3(pos_x, 0, pos_z)); 

        mSceneNode[index]->scale(10.0, 10.0	, 10.0);
        mSceneNode[index]->attachObject(mEntity[index]);
		index++;
    }
	for (int i = 0; i < 21; ++i ) {
        genNameUsingIndex("R1", index, name_en);
        genNameUsingIndex("S1", index, name_sn);
        mEntity[index] = mSceneMgr
            ->createEntity( name_en, "Barrel.mesh" );
		float pos_x = 600;
		float pos_z = -600 + 60 * i;

        mSceneNode[index] = mSceneMgr
            ->getRootSceneNode()
            ->createChildSceneNode( 
            name_sn, Vector3(pos_x, 0, pos_z)); 

        mSceneNode[index]->scale(10.0, 10.0	, 10.0);
        mSceneNode[index]->attachObject(mEntity[index]);
		index++;
    }

	//large sphere
	mLargeSphereEntity = mSceneMgr->createEntity( "l_sphere_ent", "sphere.mesh" );
	mLargeSphereEntity->setMaterialName("Examples/yellow");
	mLargeSphereSceneNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("l_sphere_n", Vector3(0,0,0));
	mLargeSphereSceneNode->attachObject(mLargeSphereEntity);


    
}

void BasicTutorial_00::createScene_00(void) 
{
	mSceneMgr = mSceneMgrArr[0];
	//mSceneMgr->setAmbientLight( ColourValue( 0.25, 0.25, 0.25 ) ); 

	mSceneMgr->setAmbientLight( ColourValue( 0.5, 0.5, 0.5 ) ); 
	//mSceneMgr->setAmbientLight( ColourValue( 1, 1, 1 ) );  
	mSceneMgr->setShadowTechnique(
		SHADOWTYPE_STENCIL_ADDITIVE); 

	Light *light;
	light = mSceneMgr->createLight("Light1"); 
	light->setType(Light::LT_POINT); 
	light->setPosition(Vector3(150, 250, 100)); 
	light->setDiffuseColour(0.3, 0.3, 0.3);		
	light->setSpecularColour(0.5, 0.5, 0.5);	

	light = mSceneMgr->createLight("Light2"); 
	light->setType(Light::LT_POINT); 
	light->setPosition(Vector3(-150, 300, 250)); 
	light->setDiffuseColour(0.25, 0.25, 0.25);		
	light->setSpecularColour(0.35, 0.35, 0.35);	

	//
	Plane plane(Vector3::UNIT_Y, 0); 
	MeshManager::getSingleton().createPlane(
		"ground", 						ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, 
		plane, 
		1500,1500, 	// width, height
		20,20, 		// x- and y-segments
		true, 		// normal
		1, 			// num texture sets
		5,5, 		// x- and y-tiles
		Vector3::UNIT_Z	// upward vector
		); 

	Entity *ent = mSceneMgr->createEntity(
		"GroundEntity", "ground"); 
	//ent->setMaterialName("Examples/BeachStones");
	//ent->setMaterialName("Examples/Rockwall");


	mSceneMgr
		->getRootSceneNode()
		->createChildSceneNode()
		->attachObject(ent); 

    ent = mSceneMgr->createEntity(
		"Background", "ground"); 
	//ent->setMaterialName("Examples/BeachStones");
	ent->setMaterialName("Examples/Rockwall");

    //Radian angle((45/180.0)*3.141592654);
    Radian angle(3.141952654/2.0);

    Vector3 axis(1.0, 0.0, 0.0);
    mSceneMgr
		->getRootSceneNode()
		->createChildSceneNode(
            Vector3(0.0, 0.0, -750.0),
            Quaternion( angle, axis))
		->attachObject(ent);

    createSpace();

    ///////////////////////
        // add your own stuff
        ///////////////////////
    
    resolveCollision();
}

void BasicTutorial_00::createScene_01(void) 
{

}

void BasicTutorial_00::createViewports(void)
{
	createViewport_00();
	createViewport_01();
}

void BasicTutorial_00::createCamera(void) {
	createCamera_00();
	createCamera_01();
	mCameraMan = mCameraManArr[0];
	//mCameraMan = mCameraManArr[1];
}

void BasicTutorial_00::createScene( void ) {
	createScene_00();
	createScene_01();
	//mSceneMgr = mSceneMgrArr[0]; // active SceneManager
	mSceneMgr = mSceneMgrArr[1]; // active SceneManager
    //
    mCamera = mCameraArr[0];
    //mCamera = mCameraArr[1];
    //
    mCameraMan->getCamera()
            ->setPosition(Vector3(-22.30,	409.24,	816.74));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.02,	-0.23,	-0.97));

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
            ->setPosition(Vector3(-22.30,	409.24,	816.74));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.02,	-0.23,	-0.97));

    }

    if (arg.key == OIS::KC_2 ) {
        mCameraMan->getCamera()
            ->setPosition(Vector3(-824.52,	468.58,	68.45));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.94,	-0.31,	-0.11));

        //-1463.00	606.45	-513.24
        //0.88	-0.47	0.10
    }


    if (arg.key == OIS::KC_3 ) {
        mCameraMan->getCamera()
            ->setPosition(Vector3(19.94,	822.63,	30.79));
        mCameraMan->getCamera()
            ->setDirection(Vector3(0.00,	-0.99,	-0.11));
        //19.94	822.63	30.79
        //0.00	-0.99	-0.11
    }
	if (arg.key == OIS::KC_7 ) {
		for(int i = 0; i<500;i++){
			mSceneNode[i]->setVisible(true);

		}
		mNumSpheres = 100;
		for(int i = mNumSpheres; i<500;i++){
			mSceneNode[i]->setVisible(false);

		}
    }
	if (arg.key == OIS::KC_8 ) {
		for(int i = 0; i<500;i++){
			mSceneNode[i]->setVisible(true);

		}
		mNumSpheres = 200;
		for(int i = mNumSpheres; i<500;i++){
			mSceneNode[i]->setVisible(false);

		}
    }
	if (arg.key == OIS::KC_9 ) {
		for(int i = 0; i<500;i++){
			mSceneNode[i]->setVisible(true);

		}
		mNumSpheres = 300;
		for(int i = mNumSpheres; i<500;i++){
			mSceneNode[i]->setVisible(false);

		}
    }
	if (arg.key == OIS::KC_0 ) {
		for(int i = 0; i<500;i++){
			mSceneNode[i]->setVisible(true);

		}
		mNumSpheres = 500;
    }

    ///////////////////////
    // add your own stuff
    ///////////////////////

    if (arg.key == OIS::KC_B ) {
		for(int i = 0; i<mNumSpheres;i++){
			float new_x = rand()%800-400;
			float new_z = rand()%800-400;
			mSceneNode[i]->setPosition(new_x,0,new_z);
		}
    }
	if (arg.key == OIS::KC_U ) {
		mMoveDirection += mMoveDirection_UP;

    }

	if (arg.key == OIS::KC_H ) {
		mMoveDirection += mMoveDirection_LEFT;

    }

	if (arg.key == OIS::KC_J ) {
		mMoveDirection += mMoveDirection_DOWN;

    }

	if (arg.key == OIS::KC_K ) {
		mMoveDirection += mMoveDirection_RIGHT;

    }

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

	if (arg.key == OIS::KC_U ) {
		mMoveDirection -= mMoveDirection_UP;

    }

	if (arg.key == OIS::KC_H ) {
		mMoveDirection -= mMoveDirection_LEFT;

    }

	if (arg.key == OIS::KC_J ) {
		mMoveDirection -= mMoveDirection_DOWN;

    }

	if (arg.key == OIS::KC_K ) {
		mMoveDirection -= mMoveDirection_RIGHT;

    }

    BaseApplication::keyReleased(arg);
	
    return flg;
}

bool BasicTutorial_00::frameStarted(const Ogre::FrameEvent& evt)
{
	bool flg = Ogre::FrameListener::frameStarted(evt);
	
    //
	


    Vector3 mdir;
    if (mMoveDirection & mMoveDirection_UP ) {
		
        mdir += Vector3(0.0, 0.0, -1.0);
		
    }
	if (mMoveDirection & mMoveDirection_DOWN ) {
		
        mdir += Vector3(0.0, 0.0, 1.0);
		
    }
	if (mMoveDirection & mMoveDirection_LEFT ) {
		
        mdir += Vector3(-1.0, 0.0, 0.0);
		
    }
	if (mMoveDirection & mMoveDirection_RIGHT ) {
		
        mdir += Vector3(1.0, 0.0, 0.0);
		
    }
	mLargeSphereSceneNode->translate(mdir);
	
	//mLargeSphereSceneNode->setPosition(new_x,0,new_z);

    ///////////////////////
    // add your own stuff
    ///////////////////////
	speed = 50.0f * evt.timeSinceLastFrame;
    resolveCollision();
    //
	
    return flg;
}

int main(int argc, char *argv[]) {
	BasicTutorial_00 app;
	app.go();  
	return 0;
}
