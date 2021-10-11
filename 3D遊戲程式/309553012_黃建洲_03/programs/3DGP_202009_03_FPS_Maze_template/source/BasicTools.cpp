#include "BasicTools.h"

namespace {
	OgreBites::SdkTrayManager* sTrayMgr;
	Ogre::RaySceneQuery *sRaySceneQuery;
	Ogre::SceneManager *sSceneMgr;
	Ogre::Camera *sCamera;
};

	using namespace Ogre;
void bt_Init(OgreBites::SdkTrayManager* a_TrayMgr, Ogre::SceneManager *a_SceneMgr, Ogre::Camera *a_Camera)
{
	sTrayMgr = a_TrayMgr;
	sSceneMgr = a_SceneMgr;
	sCamera = a_Camera;
	sRaySceneQuery = sSceneMgr->createRayQuery ( Ogre::Ray() ); 
}

void genNameUsingIndex(const Ogre::String & prefix, int index, Ogre::String &out_name)
{
	out_name= prefix + Ogre::StringConverter::toString(static_cast<int>(index));
}

void logMessage(const  Ogre::String &msg)
{
	 Ogre::LogManager::getSingletonPtr()->logMessage(msg);
}

void logMessage(const Ogre::Vector3 &v)
{
	Ogre::String msg;
	std::stringstream out0;
	std::stringstream out1;
	std::stringstream out2;
	out0 << v.x;
	out1 << v.y;
	out2 << v.z;

	msg = out0.str() + "\t\t" + out1.str() + "\t\t" + out2.str();
	
Ogre::LogManager::getSingletonPtr()->logMessage(msg);
}

bool projectScenePointOntoTerrain_PosDirection(Ogre::Vector3 &p)
{
	//Ray mRay =sTrayMgr->getCursorRay(sCamera);
	sRaySceneQuery->setRay(Ray(p, Vector3(0, 1, 0)));
	
	// Perform the scene query
	RaySceneQueryResult &result = 
		sRaySceneQuery->execute();
	RaySceneQueryResult::iterator itr = result.begin();

	bool flg = false;
	// Get the results, set the camera height
	// We are interested in the first intersection. It is ok to traverse all the results.
	if (itr != result.end() && itr->worldFragment)
	{
		p = itr->worldFragment->singleIntersection;
		
		flg = true;
	}
	return flg;
}

bool projectScenePointOntoTerrain_NegDirection(Ogre::Vector3 &p)
{
    // Add your own stuff
	//Ray mRay =sTrayMgr->getCursorRay(sCamera);
	sRaySceneQuery->setRay(Ray(p, Vector3(0, -1, 0)));
	
	// Perform the scene query
	RaySceneQueryResult &result = 
		sRaySceneQuery->execute();
	RaySceneQueryResult::iterator itr = result.begin();

	bool flg = false;
	// Get the results, set the camera height
	// We are interested in the first intersection. It is ok to traverse all the results.
	if (itr != result.end() && itr->worldFragment)
	{
		p = itr->worldFragment->singleIntersection;
		
		flg = true;
	}
	return flg;
}

Ogre::SceneNode* initParticleSystemForExplosion(
    Ogre::SceneManager *sceneMgr,
    const char *particleName
    ) {
	Ogre::SceneNode* particleNode = static_cast<SceneNode*>(
		sceneMgr->getRootSceneNode()->createChild());
	ParticleSystem* p = sceneMgr->createParticleSystem(
		particleName, "Examples/GreenyNimbus");

	ParticleEmitter *e = p->getEmitter(0);
	e->setEnabled(false);
	particleNode->attachObject(p);
	particleNode->setPosition(Vector3::ZERO);
	particleNode->setVisible(false);
    return particleNode;
}

void setOffParticleSystem(
	Ogre::SceneNode *particleNode, 
    const Ogre::String &pname, 
    const Ogre::Vector3 &pos) 
{
	particleNode->setPosition(pos);
	particleNode->setVisible(true);
	//MovableObject *s = particleNode->getAttachedObject(pname);
	MovableObject *s = particleNode->getAttachedObject(0);
	
    ParticleSystem* p =static_cast<ParticleSystem*>(s);
	if (p == NULL) return;
	p->setVisible(true);
	ParticleEmitter *e = p->getEmitter(0);
	e->setEnabled(true);
}

void setVisibleParticleSystem(
	Ogre::SceneNode *particleNode, 
    const Ogre::String &pname,
    bool flg_show) 
{
	particleNode->setVisible(flg_show);
	//MovableObject *s = particleNode->getAttachedObject(pname);
	MovableObject *s = particleNode->getAttachedObject(0);
	
    ParticleSystem* p =static_cast<ParticleSystem*>(s);
	if (p == NULL) return;
	p->setVisible(flg_show);
	ParticleEmitter *e = p->getEmitter(0);
	e->setEnabled(flg_show);
}