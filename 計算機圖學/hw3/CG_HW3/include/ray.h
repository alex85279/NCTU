#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
public:
	ray() {}
	ray(const vec3& a, const vec3& b) { O = a; D = b; }
	vec3 origin() const { return O; }
	vec3 direction() const { return D; }
	inline vec3 point_at_parameter(float t) const{ 
		/*
		To-do:
			compute the position at t
		*/


		vec3 pos;
		pos = origin() + t * direction();
		return pos; 
	}

	vec3 O; //center(origin) point
	vec3 D; //direction vector
};

#endif 
