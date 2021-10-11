#include "map_manager.h"
int MAP_MANAGER::mMaxNumObj = 64;
GAME_OBJ **MAP_MANAGER::mObj = new GAME_OBJ *[mMaxNumObj];
int MAP_MANAGER::mNumObj = 0;
SIMPLE_TERRAIN *MAP_MANAGER::mMeshMapMgr = NULL;

MAP_MANAGER::MAP_MANAGER()
{

}

void MAP_MANAGER::addObstacle(GAME_OBJ *a_GameObj)
{
	if (mNumObj > mMaxNumObj) {
		return;
	}
	mObj[mNumObj] = a_GameObj;
	mNumObj++;
}

void MAP_MANAGER::installMeshMapManager(SIMPLE_TERRAIN *a_MeshMgr)
{
	mMeshMapMgr = a_MeshMgr;
}

bool MAP_MANAGER::inside(const Vector3 &p0, const Vector3 &p1, Real t, const Plane *halfPlanes, int numHalfPlanes)
{
	bool flg = true;
	Vector3 p1p0 = p1 - p0;
	Vector3 p = p0 + t*p1p0;
	p1p0.normalise();
	p = p + p1p0*0.1;
	for (int i = 0; i < numHalfPlanes; i++) {
		Real d = p.dotProduct(halfPlanes[i].normal) - halfPlanes[i].d;
		if (d > 0) {
			flg = false;
			break;
		}
	}
	
	return flg;
}

void MAP_MANAGER::computePositionBasedOnMeshMapMgr(const Vector3 &p0, const Vector3 &p1, Vector3 &p)
{
	Vector3 diff = p1 - p0;
	float absX = diff.x > 0 ? diff.x : -diff.x;
	float absZ = diff.z > 0 ? diff.z : -diff.z;
	
	Vector3 pp = 0.5*(p1+p0);
	float v = mMeshMapMgr->getGridCellValue(p1.x, p1.z);
	float v0 = mMeshMapMgr->getGridCellValue(pp.x, pp.z);

	if (v < 0.5 && v0 < 0.5) return;
	

	if (absX < absZ) {
		p = p0 + Vector3(0, 0, diff.z); 
		v = mMeshMapMgr->getGridCellValue(p.x, p.z);
		if (v > 0.5) {
			p = p0 + Vector3(diff.x, 0, 0); 
		}
	} else {
		p = p0 + Vector3(diff.x, 0, 0); 
		v = mMeshMapMgr->getGridCellValue(p.x, p.z);
		if (v > 0.5) {
            // z component
			// Add your own stuff 
			p = p0 + Vector3(0,0,diff.z);
		}
	}
	
	
	pp = 0.5*(p+p0);
	v = mMeshMapMgr->getGridCellValue(p.x, p.z);
	v0 = mMeshMapMgr->getGridCellValue(pp.x, pp.z);

	
	if (v < 0.5 && v0 < 0.5) return;

	
	diff = p - p0;
	absX = diff.x > 0 ? diff.x : -diff.x;
	absZ = diff.z > 0 ? diff.z : -diff.z;
	if (v < 0.5) return;
	

	if (absX < absZ) {
		p = p0 + Vector3(0, 0, diff.z); 
		v = mMeshMapMgr->getGridCellValue(p.x, p.z);
		if (v > 0.5) {
			// x component
			// Add your own stuff 
			p = p0 + Vector3(diff.x,0,0);
		}
	} else {
		p = p0 + Vector3(diff.x, 0, 0); 
		v = mMeshMapMgr->getGridCellValue(p.x, p.z);
		if (v > 0.5) {
			// z component
			// Add your own stuff 
			p = p0 + Vector3(0,0,diff.z);
		}
	}
	
	if ( p == p0) {
		p.y = p1.y;
	}

	p = p0;
	
}

bool MAP_MANAGER::movePosition_Obstacles(const Vector3 &p0, const Vector3 &p1, Vector3 &p)
{
    bool flgHit = false;
    if (mNumObj >= 0) return flgHit;

	GAME_OBJ *obj = mObj[0];
	int numHalfPlanes;
	const Plane *halfPlanes = obj->getHalfPlanes(numHalfPlanes); 

	
	Real t0 = 0;
	Vector3 p1p0 = p1 - p0;
	int curNumHitPlanes  = 0;
	const Plane *hitPlane;
	
	for (int i = 0; i < numHalfPlanes ; i++) {
		const Plane *pn = &halfPlanes[i];
		Real d0 = p1p0.dotProduct(pn->normal);
		
		if (d0 <=0.0) {
			Real t = pn->d - pn->normal.dotProduct(p0);
			t = t / d0;
			
			if (t>=0 && 1.0 >= t) {
				if (!flgHit || t0 > t) {
					if (inside(p0, p1, t, halfPlanes, numHalfPlanes)) {
					flgHit = true;
					hitPlane = pn;
					t0 = t;
					break;
					}
				}
			}
		}
	}
    
    if (flgHit) {
    Real d_length = p1p0.length();
	p1p0.normalise();
	t0 *= 0.99;
	
	d_length = (1-t0)*d_length;
	Vector3 rm = p1p0*d_length; // remaining_movement
	p = p0 + t0*(p1-p0);
	Vector3 nm = hitPlane->normal;
    
	Real projected_length = rm.dotProduct(nm);
	p = p + rm - projected_length*(nm);
    }

    return flgHit;
}

bool MAP_MANAGER::movePosition(const Vector3 &p0, const Vector3 &p1, Vector3 &p)
{
	
	p = p1;
	bool flgHit = false;

    flgHit = movePosition_Obstacles(
        p0, p1, p
        );
	if (flgHit) return flgHit;

	if (!flgHit) {
		p = p1;
		//
		if (mMeshMapMgr) {
			computePositionBasedOnMeshMapMgr(p0, p1, p);
		}
		//
		return true;
	}

	//std::cout << "t0:" << t0 << std::endl;
	
    
	
		
}


Vector3 MAP_MANAGER::getGridNormalVector( float x, float z)
{
    return mMeshMapMgr->getGridNormalVector( x, z );
}