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
	robot_x = 4;
	robot_z = 3;
	for(int i = 0; i <10; i++){
		for(int j = 0; j<10 ;j++){
			map[i][j] = 0;
			if(i == 0) map[i][j] = 1;
			if(j == 0 && i != 4) map[i][j] = 1;
			if(j == 9 && i != 5) map[i][j] = 1;
			if(i == 9) map[i][j] = 1;
			if(i == 2 && j >= 2 && j <= 7) map[i][j] = 1;
			if(i == 7 && j >= 2 && j <= 7) map[i][j] = 1;
			if(i == 4 && (j == 4 || j == 1)) map[i][j] = 1;
			if(i == 5 && (j == 5 || j == 8)) map[i][j] = 1;
			if(i == 4 && j == 3) map[i][j] = 2;
			
		}
	}
	aStarFlg = false;
	openIdx = 0;
	closeIdx = 0;
	famIdx = 0;
	isWalking = false;
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




// create all spheres
// "Examples/red"
// "Examples/green"
// "Examples/blue"
// "Examples/yellow"
void BasicTutorial_00::createSpace()
{
    String name_en;
    String name_sn;
    
	for(int i = 0; i<10; i++){
		for(int j = 0; j<10; j++){
			if(map[i][j] == 1){
				genNameUsingIndex("R1",index, name_en);
				genNameUsingIndex("S1",index, name_sn);
				mEntity[index] = mSceneMgr->createEntity(name_en,"cube.mesh");
				mEntity[index]->setMaterialName("Examples/red");
				float pos_x = -500 + 100 * i;
				float pos_z = -500 + 100 * j;
				mSceneNode[index] = mSceneMgr->getRootSceneNode()->createChildSceneNode(name_sn,Vector3(pos_x,50,pos_z));
				mSceneNode[index]->attachObject(mEntity[index]);
				mSceneNode[index]->scale(0.5, 0.5, 0.5);

				index ++;
			}
			if(map[i][j] == 2){
				mEntity[index] = mSceneMgr->createEntity("robotEn","robot.mesh");
				float pos_x = -500 + 100 * i;
				float pos_z = -500 + 100 * j;
				mSceneNode[index] = mSceneMgr->getRootSceneNode()->createChildSceneNode("robotSn",Vector3(pos_x,50,pos_z));
				mSceneNode[index]->attachObject(mEntity[index]);
				mSceneNode[index]->scale(0.5, 0.5, 0.5);
				robotNode = mSceneNode[index];
				robotEnt = mEntity[index];
				
				mAnimationState = robotEnt->getAnimationState("Idle");
				mAnimationState->setLoop(true);
				mAnimationState->setEnabled(true);
;


				index ++;
			
			}
		
		}
	}

	//創造球
	for(int i = 0; i<10; i++){
		for(int j = 0; j<10; j++){
			genNameUsingIndex("BR",index, name_en);
			genNameUsingIndex("BS",index, name_sn);
			mBallEntity[i][j] = mSceneMgr->createEntity(name_en,"sphere.mesh");
			mBallEntity[i][j]->setMaterialName("Examples/blue");
			float pos_x = -500 + 100 * i;
			float pos_z = -500 + 100 * j;
			mBallNode[i][j] = mSceneMgr->getRootSceneNode()->createChildSceneNode(name_sn,Vector3(pos_x,0,pos_z));
			mBallNode[i][j]->attachObject(mBallEntity[i][j]);
			mBallNode[i][j]->scale(0.1, 0.1, 0.1);
			mBallNode[i][j]->setVisible(false);
			index ++;
		
		}
	}

	//barrels
	/*int offset = mNumSpheres;
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
	mLargeSphereSceneNode->attachObject(mLargeSphereEntity);*/


    
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


    ///////////////////////
    // add your own stuff
    ///////////////////////


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
	mAnimationState->addTime(evt.timeSinceLastFrame * 1.5f);
	if(isWalking == true){
		mAnimationState = robotEnt->getAnimationState("Walk");
		mAnimationState->setLoop(true);
		mAnimationState->setEnabled(true);
		mAnimationState->addTime(evt.timeSinceLastFrame * 1.5f);
	}
	else{
		mAnimationState = robotEnt->getAnimationState("Idle");
		mAnimationState->setLoop(true);
		mAnimationState->setEnabled(true);
		mAnimationState->addTime(evt.timeSinceLastFrame * 1.5f);
	}
    //
	if(isWalking == true && path_length > 0){
		int goal_x = final_path[0].x;
		int goal_z = final_path[0].z;
		float dx = goal_x - robot_x;
		float dz = goal_z - robot_z;
		float dt = evt.timeSinceLastFrame;
		Vector3 direction_robot = Vector3(dx,0,dz);
		robotNode->translate(direction_robot * 100 * dt);
		
		float coor_x = -500 + 100 * goal_x;
		float coor_z = -500 + 100 * goal_z;
		float dis_x = coor_x - robotNode->getPosition().x;
		float dis_z = coor_z - robotNode->getPosition().z;
		float dis = sqrt(dis_x * dis_x + dis_z + dis_z);
		robotNode->lookAt(Vector3(coor_x,robotNode->getPosition().y, coor_z), Node::TS_WORLD, Vector3::UNIT_X);
		if(dis <= 1){
			robotNode->setPosition(coor_x, robotNode->getPosition().y, coor_z);
			robot_x = goal_x;
			robot_z = goal_z;
			mBallNode[goal_x][goal_z]->setVisible(false);
			for(int i = 0; i<path_length;i++){
				
				if(i+1 < path_length){
					final_path[i] = final_path[i+1];
					
				}
				else{
					final_path[i] = mapNode();
				}
			}
			path_length--;
			if(path_length == 0){
				isWalking = false;
			}

		}
	
	}


	

    
    //
	
    return flg;
}

int main(int argc, char *argv[]) {
	BasicTutorial_00 app;
	app.go();  
	return 0;
}
bool BasicTutorial_00::aStar(){
	//將起點放入openSet
	openSet[openIdx].x = start_x;
	openSet[openIdx].z = start_z;
	//openSet[openIdx].g_score = abs(start_x - openSet[openIdx].x) + abs(start_z - openSet[openIdx].z);
	openSet[openIdx].g_score = 0;
	openSet[openIdx].h_score = abs(target_x - openSet[openIdx].x) + abs(target_z - openSet[openIdx].z);
	openSet[openIdx].f_score = openSet[openIdx].g_score + openSet[openIdx].h_score;
	openIdx++;
	//openSet[0] = mapNode();
	while(openIdx > 0){
		//求openSet中f score最小的node
		int lowest_idx;
		int lowest_fs = 10000;
		for(int i = 0; i<openIdx; i++){
			if(openSet[i].f_score < lowest_fs) {
				lowest_fs = openSet[i].f_score;
				lowest_idx = i;
			}
		}
		//用nowNode存下來，把得到的node從openSet中移除，加入closeSet
		mapNode nowNode = openSet[lowest_idx];
		if(openSet[lowest_idx].x == target_x && openSet[lowest_idx].z == target_z){
			std::cout<<"reach goal"<<std::endl;
			return true;
		}
		closeSet[closeIdx++] = openSet[lowest_idx];
		for(int i = lowest_idx; i<openIdx;i++){
			if(i+1 < openIdx) openSet[i] = openSet[i+1];
			else openSet[i] = mapNode();
		}
		openIdx--;
		for(int i = 0; i<4 ;i++){
			int neibor_x;
			int neibor_z;
			if(i == 0){
				neibor_x = nowNode.x+1;
				neibor_z = nowNode.z;
			}
			else if(i == 1){
				neibor_x = nowNode.x - 1;
				neibor_z = nowNode.z;
			}
			else if(i == 2){
				neibor_x = nowNode.x;
				neibor_z = nowNode.z + 1;
			}
			else if(i == 3){
				neibor_x = nowNode.x;
				neibor_z = nowNode.z - 1;
			}
			std::cout<<"neighbor"<<i<<": "<<neibor_x << " "<< neibor_z<<std::endl;
			//檢查neibor可不可以走
			if(neibor_x<=0 || neibor_x >=9 || neibor_z <= 0 || neibor_z >= 9 || map[neibor_x][neibor_z] != 0){
				continue;
			}

			//檢查鄰居是否有在closeSet裡面
			int in_closeSet = 0;
			for(int j = 0; j<closeIdx; j++){
				if(closeSet[j].x == neibor_x && closeSet[j].z == neibor_z){
					in_closeSet = 1;
					break;
				}
			}
			if(in_closeSet == 1){
				continue;	
			}
			int tmp_g_score = nowNode.g_score + 1;
			int real_g_score = abs(start_x - neibor_x) + abs(start_z - neibor_z);
			int in_openSet = 0;
			int tmp_is_better = 0;
			//檢查neighbor是否在openSet裡面
			for(int j = 0; j<openIdx;j++){
				if(openSet[j].x == neibor_x && openSet[j].z == neibor_z){
					in_openSet = 1;
					break;
				}
			}
			if(in_openSet == 0){
				tmp_is_better = 1;
			}
			else if(tmp_g_score < real_g_score){
				tmp_is_better = 1;
			}
			else{
				tmp_is_better = 0;
			}
			if(tmp_is_better == 1){
				//儲存path訊息
				fam[famIdx].parent_x = nowNode.x;
				fam[famIdx].parent_z = nowNode.z;
				fam[famIdx].child_x = neibor_x;
				fam[famIdx].child_z = neibor_z;
				famIdx++;
				//將neibor加入openset
				openSet[openIdx].x = neibor_x;
				openSet[openIdx].z = neibor_z;
				openSet[openIdx].g_score = tmp_g_score;
				openSet[openIdx].h_score = abs(target_x - openSet[openIdx].x) + abs(target_z - openSet[openIdx].z);
				openSet[openIdx].f_score = openSet[openIdx].g_score + openSet[openIdx].h_score;
				openIdx++;
			}




		}



	}
	return false;
}

bool BasicTutorial_00::mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id ){
	if(id == OIS::MB_Left){
		for(int i = 0; i<10; i++){
			for(int j = 0; j<10; j++){
				mBallNode[i][j]->setVisible(false);
			
			}
		}
		Ray mRay = mTrayMgr->getCursorRay(mCamera);
		std::pair<bool,Real> result = mRay.intersects(mPlane);
		if(result.first == true){
			target_pos = mRay.getPoint(result.second);
			float fx = target_pos.x/100;
			float fz = target_pos.z/100;
			int ix = (fx >= 0)?int(fx + 0.5) : int(fx - 0.5);
			int iz = (fz >= 0)?int(fz + 0.5) : int(fz - 0.5);
			ix = ix + 5;
			iz = iz + 5;
			target_x = ix;
			target_z = iz;
			std::cout<<target_pos<<std::endl;
			std::cout<<ix<<" "<<iz<<std::endl;
			if(ix > 0 && ix < 9 && iz > 0 && iz < 9 && map[ix][iz]!=1 && map[ix][iz] != 2){
				aStarFlg = true;
				std::cout<<"legal target"<<std::endl;
				float rfx = robotNode->getPosition().x/100;
				float rfz = robotNode->getPosition().z/100;
				int rix = (rfx >= 0)?int(rfx + 0.5) : int(rfx - 0.5);
				int riz = (rfz >= 0)?int(rfz + 0.5) : int(rfz - 0.5);
	
				rix = rix + 5;
				riz = riz + 5;
				if(isWalking == true){
					start_x = final_path[0].x;
					start_z = final_path[0].z;
				}
				else{
					start_x = rix;
					start_z = riz;
				}
				bool isPath = aStar();
				if(isPath){
					std::cout<<"Reconstruct Path:"<<std::endl;
					int path_x = target_x;
					int path_z = target_z;
					std::cout<<path_x<<" "<<path_z<<std::endl;
					mapNode tmp_path[100];
					path_length = 0;
					tmp_path[path_length].x = target_x;
					tmp_path[path_length].z = target_z;
					path_length++;
					while(!(path_x == start_x && path_z == start_z)){
						for(int i = 0; i<famIdx;i++){
							if(path_x == fam[i].child_x && path_z == fam[i].child_z){
								path_x = fam[i].parent_x;
								path_z = fam[i].parent_z;
								std::cout<<path_x<<" "<<path_z<<std::endl;
								if(!(path_x == start_x && path_z == start_z) && !(path_x == target_x && path_z == target_z)){
									tmp_path[path_length].x = path_x;
									tmp_path[path_length].z = path_z;
									path_length++;
								}
								break;
							}
						}
					}
					
					//建立完路線和顯示球
					int j = 0;
					//檢測是否已經有上一條路線:
					if(isWalking == true){
						j = 1;
					}
					std::cout<<"final_path"<<std::endl;
					for(int i = path_length - 1; i >=0; i--){
						final_path[j] = tmp_path[i];
						
						
						j++;

					}
					if(isWalking == true){
						path_length +=1;
					}
					for(int i = 0; i<path_length;i++){
						std::cout<<final_path[i].x << " "<< final_path[i].z<<std::endl;
						mBallNode[final_path[i].x][final_path[i].z]->setVisible(true);
					}
					isWalking = true;



				}
				else{
					std::cout<<"There is no Path"<<std::endl;
				}

				openIdx = 0;
				closeIdx = 0;
				famIdx = 0;
			}
			

		}	
		
	}

	return BaseApplication::mousePressed( arg, id );
}