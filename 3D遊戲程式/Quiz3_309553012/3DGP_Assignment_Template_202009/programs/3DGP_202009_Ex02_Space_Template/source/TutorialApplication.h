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
My Email: aaa@cs.nctu.edu.tw
\n Date: 2019/10/28

This is an assignment of 3D Game Programming

*/

#ifndef __BasicTutorial_00_h_
#define __BasicTutorial_00_h_

#include "BaseApplication.h"

using namespace Ogre;
struct family{
	int parent_x;
	int parent_z;
	int child_x;
	int child_z;
};
struct mapNode{
	int x;
	int z;
	int g_score;
	int h_score;
	int f_score;
};
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
	mapNode openSet[100];
	int openIdx;

	mapNode closeSet[100];
	int closeIdx;

	family fam[100];
	int famIdx;
	bool aStar();
	void createViewport_00(void);
	void createViewport_01(void);
	//
	void createCamera_00();
	void createCamera_01();

	void createScene_00();
	void createScene_01();
    bool keyPressed( const OIS::KeyEvent &arg );
    bool keyReleased( const OIS::KeyEvent &arg );
	virtual bool mousePressed( const OIS::MouseEvent &arg, OIS::MouseButtonID id );
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
	int map[10][10];
	SceneNode *mRobotNode;
	Plane mPlane;
	Vector3 target_pos;
	int target_x;
	int target_z;
	bool aStarFlg;
	int start_x;
	int start_z;
	SceneNode *robotNode;
	Entity *robotEnt;
	mapNode final_path[100];
	int path_length;
	SceneNode *mBallNode[10][10];
    Entity *mBallEntity[10][10];
	bool isWalking;
	int robot_x;
	int robot_z;
	AnimationState *mAnimationState;
};

#endif // #ifndef __BasicTutorial_00_h_