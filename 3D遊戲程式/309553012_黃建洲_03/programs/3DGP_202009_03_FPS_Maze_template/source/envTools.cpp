/*
This is a game demo written by Wingo Sai-Keung Wong.
OGRE is employed for graphics rendering.
CEGUI is employed for GUI.
Date: Dec 2006 - 2009
Email: wingo.wong@gmail.com

All rights reserved. 2009
*/
#include "envtools.h"

namespace {
	SceneManager *mSceneMgr = NULL;
	RaySceneQuery* raySceneQuery = NULL;
};

void initEnvTools(SceneManager *m)
{
	mSceneMgr = m;
	raySceneQuery = mSceneMgr->createRayQuery(
			Ray(Vector3::ZERO, Vector3::NEGATIVE_UNIT_Y));
}

bool clampToEnvironment(const Vector3& cur_pos, Real offset, Vector3& new_pos)
{
    if (raySceneQuery==0) return false;

	static Ray updateRay;
	updateRay.setOrigin(cur_pos);
	updateRay.setDirection(Vector3::NEGATIVE_UNIT_Y);
	raySceneQuery->setRay(updateRay);
	RaySceneQueryResult& qryResult = raySceneQuery->execute();
	RaySceneQueryResult::iterator i = qryResult.begin();

	//
	bool flg = false;
	raySceneQuery->setRay(updateRay);
	qryResult =raySceneQuery->execute();
	i = qryResult.begin();
	Real i_y = 0.0;
	//
	new_pos = cur_pos;
	//
	if (i != qryResult.end() && i->worldFragment)
	{
		i_y = i->worldFragment->singleIntersection.y;

		new_pos.y = i_y + offset;
		flg = true;
	}

	if (flg) return flg;
    // Handle up direction
	updateRay.setDirection(-Vector3::NEGATIVE_UNIT_Y);
    // Add your own stuff
	
	
		


	return flg;
}


/*
bool checkFireHitTarget_SceneObj_Intersect(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance) 
{
#ifndef ENABLE_COLLISION_DETECTION
	return false;
#endif
	bool flg_hit = false;
	//return flg_hit;
	if (mSceneMgr == NULL) return flg_hit;

	static Ray updateRay;
	updateRay.setOrigin(start_pos);
	updateRay.setDirection(fire_d);
	raySceneQuery->setRay(updateRay);

	RaySceneQueryResult& qryResult = raySceneQuery->execute();
	RaySceneQueryResult::iterator i = qryResult.begin();

	for (; i != qryResult.end(); i++) {	
		if (i->movable == NULL) continue;

		*hit_target = i->movable->getParentSceneNode();
		DEBUG_LOG_MSG_POINTER(*hit_target);

		if (i->movable->getQueryFlags() == FIREHIT_QUERY_MASK) {

			//
			Real valid_distance = start_pos.distance(end_pos);
			hit_distance = i->distance;
			if (valid_distance >= hit_distance) {
				Vector3 p = start_pos+i->distance*fire_d;
				Vector3 pos = (*hit_target)->getPosition();
				hit_target_world_pos = p;

				Quaternion q = (*hit_target)->getOrientation();
				hit_target_pos = q.Inverse()*(p - pos);

				flg_hit = true;
				break;
			}
		}
	}
	//
	return flg_hit;
}

bool checkFireHitTarget_SceneObj_Sphere(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance) 
{
#ifndef ENABLE_COLLISION_DETECTION
	return false;
#endif
	bool flg_hit = false;
	//return flg_hit;
	if (mSceneMgr == NULL) return flg_hit;
	//static Axis-aligned;
	Vector3 c = (start_pos+end_pos)*0.5;
	Real d = (start_pos.distance(end_pos))*0.5;
	const Sphere sphere(c, d);
	static SphereSceneQuery *query = mSceneMgr->createSphereQuery(sphere, FIREHIT_QUERY_MASK);
	query->setSphere(sphere);
	query->setQueryMask(FIREHIT_QUERY_MASK);
	SceneQueryResult& results = query->execute();

	SceneQueryResultMovableList::iterator i = results.movables.begin();

	//RaySceneQueryResult::iterator j = results.begin();
	//
	for (; i != results.movables.end(); i++) {
		if ((*i)->getQueryFlags() == FIREHIT_QUERY_MASK) {

			*hit_target = (*i)->getParentSceneNode();
			DEBUG_LOG_MSG_POINTER(*hit_target);


			Vector3 pos = (*hit_target)->getPosition();
			hit_target_world_pos = start_pos;
			hit_target_pos = start_pos - pos;
			Quaternion q = (*hit_target)->getOrientation();
			hit_target_pos = q.Inverse()*hit_target_pos;
			hit_distance = 0;
			flg_hit = true;
			break;
		}
	}
	//
	return flg_hit;
}

bool checkFireHitTarget_SceneObj(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance)
{
	
	return checkFireHitTarget_SceneObj_Sphere(
		fire_d, 
		start_pos, 
		end_pos, 
		hit_target, 
		hit_target_pos, 
		hit_target_world_pos, 
		hit_distance);
}

bool checkFireHitTarget_World(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance) {
#ifndef ENABLE_COLLISION_DETECTION
	return false;
#endif
#ifdef DISABLE_COLLISION_DETECTION_WORLD
	return false;
#endif
	static Ray updateRay;
	updateRay.setOrigin(start_pos);
	updateRay.setDirection(fire_d);
	raySceneQuery->setRay(updateRay);

	RaySceneQueryResult& qryResult = raySceneQuery->execute();
	RaySceneQueryResult::iterator i = qryResult.begin();

	bool flg_hit = false;
	if (i != qryResult.end() && i->worldFragment) {
		Vector3 p;
		p.x = i->worldFragment->singleIntersection.x;
		p.y = i->worldFragment->singleIntersection.y;
		p.z = i->worldFragment->singleIntersection.z;
		Real valid_distance = start_pos.distance(end_pos);
		hit_distance = i->distance;
		if (valid_distance >= hit_distance) {
			hit_target_pos = p;
			hit_target_world_pos = p;
			flg_hit = true;
		}

	}
	return flg_hit;
}

bool checkFireHitTarget(const Vector3 &fire_d, const Vector3 &start_pos, const Vector3 &end_pos, SceneNode **hit_target, Vector3 &hit_target_pos, Vector3 &hit_target_world_pos, Real &hit_distance) {
#ifndef ENABLE_COLLISION_DETECTION
	return false;
#endif
	bool flg_hit = false;
	Vector3 p0, p1;
	Vector3 wp0, wp1;
	Real d0, d1;
	int flg0 = checkFireHitTarget_World(fire_d, start_pos, end_pos, hit_target, p0, wp0, d0);
	int flg1 = checkFireHitTarget_SceneObj(fire_d, start_pos, end_pos, hit_target, p1, wp1, d1);
	if (flg0 || flg1) {
		flg_hit = true;
		int type = 0;
		if (flg0 && !flg1) {
			type = 1;
		}
		if (!flg0 && flg1) {
			type = 2;
		}

		if (flg0 && flg1) {
			if (d0>d1) {
				type = 2;
			} else {
				type = 1;
			}
		}

		if (type == 1) {
			hit_target_pos = p0;
			hit_target_world_pos = wp0;
			hit_distance = d0;
		}
		if (type == 2) {
			hit_target_pos = p1;
			hit_target_world_pos = wp1;
			hit_distance = d1;
		}
	}
	return flg_hit;
}
*/