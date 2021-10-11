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
\n Date: 2020/09/25

This is an assignment of 3D Game Programming
*/

////////////////////////////////////////
// You can delete or add some functions to do the assignment.
////////////////////////////////////////

#ifndef __BasicTutorial_00_h_
#define __BasicTutorial_00_h_

#include "BaseApplication.h"

class BasicTutorial_00 : public BaseApplication
{
public:
	BasicTutorial_00(void);
    virtual void chooseSceneManager(void);
    virtual void createCamera(void);
	virtual void createViewports(void);
	virtual void createScene(void);
	
	
    /*!
	\brief Define the motion of the items before the frame is rendered
		
	Small penguin: Reset its position to make it move along the circle with large penguin as the center.
	
	The angular speed increase by time. After the small penguin finish one round, the angular speed will be reset and the direction of the small penguin will be opposite.

	Large penguin: Reset its direction to make it always look at the small penguin.
	

	\return Return the flag of frame started
	*/
    virtual bool frameStarted(const Ogre::FrameEvent& evt);
    //
    // Add your own stuff.
    //
	float mAngle;
	float mAngularSpeed;
	int reverse;
	bool toggle;
	int place_idx;
protected:
	/*!
	\brief Create a viewport

	Create a viewport for the entire screen.

	\return No return value
	*/
	void createViewport_00(void);
	/*!
	\brief Create a viewport for 2nd Scene Manager

	Create the top level viewport for the screen in the corner.

	\return No return
	*/
	void createViewport_01(void);
	//
	/*!
	\brief Create a camera item for the 1st Scene Manager

	Describe the property of the camera(Position, LookAt, ClipDistance etc.)

	Create a camera man and attach the camera for Manager

	\return No return value
	*/
	void createCamera_00();

	/*!
	\brief Create a camera item for the 2nd Scene Manager

	Describe the property of the camera(Position, LookAt, ClipDistance etc.)

	Create a camera man and attach the camera for Manager

	\return No return value
	*/
	void createCamera_01();

	/*!
	\brief Create a scene for 1st Scene Manager

	What to do:

	1. Set ambient light and shadow technique.

	2. Create plane for placing items (Ensure that plane won't cast shadow) and attach it to child node.

	3. (Large Penguin)Create a penguin entity, attach to child node, set propertites, scale it to make it become larger, rotate it to make it face to camera.

	4. (Small Penguin)Same as 4., but without scaling, and make it face to the large penguin.

	5. Create several nodes with cube entity, place and scale them according to the provided function.

	6. Create two light item to observe the performance of the shadow.

	\return No return value
	*/

	void createScene_00();

	/*!
	\brief Create a scene for 2nd Scene Manager

	What to do: (just a simple scene)

	1. Set ambient light and shadow technique.

	2. Create plane for placing items (Ensure that plane won't cast shadow) and attach it to child node.

	3. Create a node with cube entity and place it to the middle of the ground.

	4. Create one light item to observe the performance of the shadow.

	\return No return value
	*/
	
	void createScene_01();
	/*!
	\brief Set the callback function according to the keyboard input

	Functions description:
	
	1. key '1' ~ '5': From camera man get the camera, then reset the position and direction of the camera in order to get desired view.
	
	2. key 'B': According to the viewport-position state, set the position of the top viewport to top-left, top-right, bottem-right, bottom-left(in order) of the screen.
	
	3. key 'M': Reset the two viewports, make the viewport of 2nd camera become the bottem level and the viewport of 1st camera become the top level(top-left corner). Set the viewport-position to top-left.
	
	4. key 'N': Reset the two viewports, make the viewport of 1nd camera become the bottem level and the viewport of 2nd camera become the top level(top-right corner). Set the viewport-position to top-right.
	
	5. key 'P': Press to start or stop the animation.
	
	\return Flag of key pressed, must be true.
	*/
    bool keyPressed( const OIS::KeyEvent &arg );
    bool keyReleased( const OIS::KeyEvent &arg );
    //
    // Add your own stuff.
    //
protected:
    Ogre::Viewport* mViewportArr[8];
	Ogre::Camera* mCameraArr[8];
	Ogre::SceneManager* mSceneMgrArr[8];
	OgreBites::SdkCameraMan* mCameraManArr[8];
    //
    // Add your own stuff.
    //
};


#endif // #ifndef __BasicTutorial_00_h_

////////////////////////////////////////
// DO NOT DELETE THIS LINE: 2020/09/25:ABCD 3D Game Programming
////////////////////////////////////////