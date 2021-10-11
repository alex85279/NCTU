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
	mode = 0;
	arrival_flag = 0;
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
	mCamera->setPosition(Ogre::Vector3(0,1200,0.01));
	mCamera->lookAt(Ogre::Vector3(0,0,0));
	mCamera->setNearClipDistance(5);
	mCameraManArr[0] = new OgreBites::SdkCameraMan(mCamera);   // create a default camera controller

	tmpCamera = mCameraArr[1] = mSceneMgr->createCamera("trashCam");
	tmpCamera->setPosition(Ogre::Vector3(120,300,600));
	tmpCamera->lookAt(Ogre::Vector3(120,0,0));
	tmpCamera->setNearClipDistance(5);
	mCameraManArr[2] = new OgreBites::SdkCameraMan(tmpCamera);   // create a default camera controller

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
Vector3 BasicTutorial_00::randomCircle(Vector3 pos){
	int rnd_x = rand() % 1000 - 500;
	int rnd_z = rand() % 1000 - 500;
	Vector3 targetDirection(rnd_x,0,rnd_z);
	targetDirection.normalise();
	Vector3 targetPos = pos + targetDirection * 100;
	return targetPos;

}
void BasicTutorial_00::createSpace()
{
    String name_en;
    String name_sn;
	
	ent = mSceneMgr
			->createEntity("haha", "sphere.mesh" );
	node = mSceneMgr->getRootSceneNode()->createChildSceneNode("hi");
	node->attachObject(ent);
    node->scale(0.15,0.15,0.15);
	node->setVisible(false);
    mNumSpheres = 100;
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
		now_dir[index] = Vector3(1,0,0);
		

		//std::cout<<"origin pos: "<<mSceneNode[index]->getPosition()<<std::endl;
		//std::cout<<"WanderTarget pos: "<<wanderTarget[index]<<std::endl;
		//std::cout<<"distance "<<mSceneNode[index]->getPosition().distance(wanderTarget[index])<<std::endl;


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
	mPlane = plane;
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
	mCameraMan = mCameraManArr[2];
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
	if (arg.key == OIS::KC_W ) {
		arrival_flag = 0;
		chase_flag = 0;
		mode = 1;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(false);
		}
		mNumSpheres = 100;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(true);
			float rand_x = rand()%800 - 400;
			float rand_z = rand()%800 - 400;	
			mSceneNode[i]->setPosition(Vector3(rand_x,0,rand_z));
		}

	
	}
	if (arg.key == OIS::KC_A || (arg.key == OIS::KC_SPACE && mode == 3) ) {
		arrival_flag = 0;
		chase_flag = 0;
		mode = 2;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(false);
		}
		mNumSpheres = 100;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(true);
			float rand_x = rand()%800 - 400;
			float rand_z = rand()%800 - 400;	
			mSceneNode[i]->setPosition(Vector3(rand_x,0,rand_z));
		}
	}
	else if (arg.key == OIS::KC_S || (arg.key == OIS::KC_SPACE && mode == 2)) {
		arrival_flag = 0;
		chase_flag = 0;
		mode = 3;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(false);
		}
		mNumSpheres = 100;
		for(int i = 0; i<mNumSpheres; i++){
			mSceneNode[i]->setVisible(true);
			float rand_x = rand()%800 - 400;
			float rand_z = rand()%800 - 400;
			if(i == 0){
				rand_x = 0;
				rand_z = 0;
			}

			mSceneNode[i]->setPosition(Vector3(rand_x,0,rand_z));
		}
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
void BasicTutorial_00::wandering(float dt){
	for(int i = 0; i< mNumSpheres; i++){
		Vector3 dir;
		Vector3 circle_pos = mSceneNode[i]->getPosition() + now_dir[i] * 500;
		Vector3 target_pos = randomCircle(circle_pos);
		dir = target_pos - mSceneNode[i]->getPosition();
		dir.normalise();
		now_dir[i] = dir;
		
		mSceneNode[i]->translate(dir * 100 * dt);
		//node->setPosition(randomCircle(circle_pos));
	
	}

}
void BasicTutorial_00::chase(float dt){
	Vector3 dir = chase_target - mSceneNode[0]->getPosition();
	dir.normalise();
	float dis = chase_target.distance(mSceneNode[0]->getPosition());
	Vector3 original_pos = mSceneNode[0]->getPosition();
	if(dis > 100){
		mSceneNode[0]->translate(dir * 100 * dt);
	}
	else{
		mSceneNode[0]->translate(dir * dis * dt);
	}

	if(dis <= 1){
		mSceneNode[0]->setPosition(chase_target);
		chase_flag = 0;
	}
	for(int i = 1; i<mNumSpheres;i++){
		Vector3 dir1 = original_pos - mSceneNode[i]->getPosition();
		dir1.normalise();
		float dis1 = mSceneNode[0]->getPosition().distance(mSceneNode[i]->getPosition());
		mSceneNode[i]->translate(dir1 * 100 * dt);
		if(dis1 <= 40){
			float rand_x = rand()%800 - 400;
			float rand_z = rand()%800 - 400;	
			mSceneNode[i]->setPosition(Vector3(rand_x,0,rand_z));
		}
	}
}
void BasicTutorial_00::arrival(float dt){
	
	for(int i = 0; i<mNumSpheres; i++){
		Vector3 dir = arrival_target - mSceneNode[i]->getPosition();
		dir.normalise();
		float dis = arrival_target.distance(mSceneNode[i]->getPosition());
	
		if(dis > 100){
			mSceneNode[i]->translate(dir * 100 * dt);
		}
		else{
			mSceneNode[i]->translate(dir * dis * dt);
		}

		if(dis <= 20){

			float rand_x = rand()%800 - 400;
			float rand_z = rand()%800 - 400;	
			mSceneNode[i]->setPosition(Vector3(rand_x,0,rand_z));
		}
	}
	
}
bool BasicTutorial_00::frameStarted(const Ogre::FrameEvent& evt)
{
	bool flg = Ogre::FrameListener::frameStarted(evt);
	float dt = evt.timeSinceLastFrame;
    //
	if(mode == 1) wandering(dt);
	if(mode == 2 && arrival_flag == 1) arrival(dt);
	if(mode == 3) chase(dt);
    
	
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
bool BasicTutorial_00::mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id ){
	
	Ray mRay = mTrayMgr->getCursorRay(mCamera);
	std::pair<bool,Real> result = mRay.intersects(mPlane);
	if(result.first == true){
		if(mode == 2){
			arrival_target = mRay.getPoint(result.second);
			arrival_flag = 1;
		}
		if(mode == 3){
			chase_target = mRay.getPoint(result.second);
			chase_flag = 1;
		}
	}	
	return BaseApplication::mousePressed( arg, id );
}