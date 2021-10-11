/*
This is a game demo written by Sai-Keung Wong.
OGRE is employed for graphics rendering.
Date: Dec 2006 - Nov. 2020
Email: wingo.wong@gmail.com

*/

#include <string>
#include "SpecialEffectManager.h"
#include "BasicTools.h"

const std::string particleSystemNames[] = {
    "Examples/GreenyNimbus",
    //"Examples/Rain",
    "Examples/Aureola",
    "Examples/Snow",
    //"Examples/Smoke"
};

SpecialEffectManager::SpecialEffectManager()
{
    mSceneMgr = 0;
    mCurrentIndex = 0;
    mNumParticleSystems = 0;
    mParticleSystemNames = 0;
}

SpecialEffectManager::SpecialEffectManager(
    SceneManager *sceneMgr
    )
{
    mSceneMgr = sceneMgr;
    mCurrentIndex = 0;
    mNumParticleSystems = sizeof(particleSystemNames)/sizeof(std::string);
    mParticleSystemNames = (std::string*)particleSystemNames;

    mParticleNodes = new SceneNode*[mNumParticleSystems];
        mParticleEmitters  = new ParticleEmitter*[mNumParticleSystems];
    
}

SpecialEffectManager::SpecialEffectManager(
    SceneManager *sceneMgr,
    const std::string *particleSystemNames,
    int numParticleSystems
    )
{
    mSceneMgr = sceneMgr;
    mCurrentIndex = 0;

    mNumParticleSystems = numParticleSystems;
    mParticleSystemNames = particleSystemNames;

    mParticleNodes = new SceneNode*[mNumParticleSystems];
        mParticleEmitters  = new ParticleEmitter*[mNumParticleSystems];
}

    SpecialEffectManager::SpecialEffectManager(
        SceneManager *sceneMgr,
        int numParticleSystems
        )
    {
        mSceneMgr = sceneMgr;
        mNumParticleSystems = numParticleSystems;
        mParticleNodes = new SceneNode*[mNumParticleSystems];
        mParticleEmitters  = new ParticleEmitter*[mNumParticleSystems];
    }


void SpecialEffectManager::createParticleSystem(int index)
{
    int i = index;
    int numPNames = mNumParticleSystems;
    int pTypeIndex = index%numPNames;

    mParticleNodes[i] = static_cast<SceneNode*>(
		mSceneMgr->getRootSceneNode()->createChild());

    String name;
		genNameUsingIndex("ps", i, name);

	ParticleSystem* p = mSceneMgr->createParticleSystem(
		name, mParticleSystemNames[pTypeIndex]);

	ParticleEmitter *e = ((ParticleSystem*)p)->getEmitter(0);
    mParticleEmitters[i] = e;
	e->setEnabled(false);
	mParticleNodes[i]->attachObject(p);
	mParticleNodes[i]->setPosition(Vector3::ZERO);
	mParticleNodes[i]->setVisible(false);
}

void SpecialEffectManager::init()
{
    for (int i =0; i < mNumParticleSystems; ++i) {
        createParticleSystem(i);
    }
}

void SpecialEffectManager::setOffParticleSystem(
    int pIndex, 
    const Ogre::Vector3 &pos)
{
    int index = pIndex;
    SceneNode *fNode = mParticleNodes[index];

    fNode->setPosition(pos);
	fNode->setVisible(true);

    //p->setVisible(true);
	ParticleEmitter *e = mParticleEmitters[index];
	e->setEnabled(true);
    
}

int SpecialEffectManager::setOffParticleSystem(const Ogre::Vector3 &pos)
{
    int index = mCurrentIndex;
    SceneNode *fNode = mParticleNodes[mCurrentIndex];

    fNode->setPosition(pos);
	fNode->setVisible(true);

    //p->setVisible(true);
	ParticleEmitter *e = mParticleEmitters[mCurrentIndex];
	//e->setEnabled(true);
    
    mCurrentIndex = (mCurrentIndex+1)%mNumParticleSystems;
    return index;
}

void SpecialEffectManager::disableAllParticleSystems()
{
    for (int i =0; i < mNumParticleSystems; ++i) {
             mParticleNodes[i]->setVisible(false);
        ParticleEmitter *e = mParticleEmitters[i];
	e->setEnabled(false);
    }
}

    void SpecialEffectManager::enableAllParticleSystems() {
         for (int i =0; i < mNumParticleSystems; ++i) {
             mParticleNodes[i]->setVisible(true);
        ParticleEmitter *e = mParticleEmitters[i];

	e->setEnabled(true);
    }
    }
