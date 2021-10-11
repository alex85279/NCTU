#include <string>
#include "WeaponParticleSystemManager.h"

const std::string particleSystemNames[] = {
    "Examples/Smoke",
    "Examples/upSmoke",
    "Examples/Smoke",
    "Examples/upSmoke",
    "Examples/Smoke",
    "Examples/upSmoke",
    "Examples/Smoke"
    //"Examples/Smoke"
};

WeaponParticleSystemManager::WeaponParticleSystemManager()
{
}

WeaponParticleSystemManager::WeaponParticleSystemManager(
    SceneManager *sceneMgr
    ): SpecialEffectManager(
    sceneMgr,
    particleSystemNames,
    sizeof(particleSystemNames)/sizeof(std::string)
    )
{

}

int WeaponParticleSystemManager::getFreeParticleSystemIndex() const
{
    int index = mCurIndex;
    mCurIndex = (mCurIndex+1)%mNumParticleSystems;

    return index;
}

void WeaponParticleSystemManager::play(int pIndex, const Vector3 &p)
{
    setOffParticleSystem(pIndex, p);
}

// cycle to reuse particle systems.
void WeaponParticleSystemManager::play(const Vector3 &p)
{
    //setOffParticleSystem(mCurIndex, p);
    //mCurIndex = (mCurIndex+1)%mNumParticleSystems;
}


