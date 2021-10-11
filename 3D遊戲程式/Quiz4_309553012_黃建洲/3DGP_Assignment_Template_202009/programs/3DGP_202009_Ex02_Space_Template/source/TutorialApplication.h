//
// 3D Game Programming
// NCTU
// Instructor: SAI-KEUNG WONG
//
/*!
\brief 3D Game Programming
\n
My Name: Chien-Chou Wong
\n
My ID: 309553012
\n
My Email: alexwong85279@gmail.com
\n Date: 2019/10/28

This is an assignment of 3D Game Programming

*/

#ifndef __BasicTutorial_00_h_
#define __BasicTutorial_00_h_

#include "BaseApplication.h"

using namespace Ogre;

class BasicTutorial_00 : public BaseApplication
{
public:
	BasicTutorial_00(void);
	virtual void createViewports(void);
	virtual void createScene(void);
	virtual void createCamera(void);
	virtual void chooseSceneManager(void);
    //
	virtual bool frameStarted(const Ogre::FrameEvent& evt);
    //
protected:
	/*!
	\brief Create a viewport

	Create a viewport for the entire screen.

	\return The sum of two integers.
	*/
	Vector3 randomCircle(Vector3 item);
	Vector3 wanderTarget[100];
	Vector3 now_dir[100];
	Entity *ent;
	SceneNode* node;
	int mode;
	int arrival_flag;
	Vector3 arrival_target;
	int chase_flag;
	Vector3 chase_target;
	Plane mPlane;
	virtual bool mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id );
	void wandering(float dt);
	void arrival(float dt);
	void chase(float dt);
	void createViewport_00(void);
	void createViewport_01(void);
	//
	void createCamera_00();
	void createCamera_01();

	void createScene_00();
	void createScene_01();
    bool keyPressed( const OIS::KeyEvent &arg );
    bool keyReleased( const OIS::KeyEvent &arg );
    void createSpace();

    void resolveCollisionSmallSpheres();
    void resolveCollisionLargeSphere();
    void resolveCollision(
    SceneNode *nodeA, SceneNode *nodeB,
    float rA, float rB);
    void resolveCollision(
    SceneNode *nodeA, SceneNode *nodeB,
    float rA, float rB, float wA, float wB);
    void resolveCollision();
    void resolveCollisionPair(
    int robotA, int robotB, float rA, float rB);

    void reset();
protected:
    Ogre::Viewport* mViewportArr[8];
	Ogre::Camera* mCameraArr[8];
	Ogre::Camera* tmpCamera;
	Ogre::SceneManager* mSceneMgrArr[8];
	OgreBites::SdkCameraMan* mCameraManArr[8];
    //
    bool mFlg_Penguin_Jump;
    bool mFlg_Penguin_Dir;
    //
    bool mFlg_Root;
    bool mFlg_Root_Dir;
        bool mFlg_Arm;
    bool mFlg_Arm_Dir;
    //
    int mMoveDirection;
    SceneNode *mLargeSphereSceneNode;
    Entity *mLargeSphereEntity;

    int mNumSpheres;
    SceneNode *mSceneNode[1024];
    Entity *mEntity[1024];
	int index;
	float speed;

};

#endif // #ifndef __BasicTutorial_00_h_