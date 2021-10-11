#ifndef GEOH
#define GEOH
#include "ray.h"
using namespace std;
class material;
typedef struct hit_record {
	float t;
	vec3 p;
	vec3 nv;

}hit_record;

class hitable { //geometry parent class
public:
	virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const = 0;
};

class sphere : public hitable {
public:
	sphere() {}
	sphere(vec3 c, float r,vec3 _kd=vec3(1.0,1.0,1.0), float w_ri = 0.0f, float w_ti = 0.0f) : 
		center(c), radius(r), kd(_kd), w_r(w_ri), w_t(w_ti) {};
	virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	vec3 center;
	float radius;
	vec3 kd;
	float w_r; //reflected
	float w_t; //transmitted
};

bool sphere::hit(const ray &r, float tmin, float tmax, hit_record & rec) const {
	/*
	To-do:
		compute whether the ray intersect the sphere
	*/
	
	vec3 o = r.origin();
	vec3 d = r.direction();
	vec3 ce = center;
	
	float rc = radius;
	float A = dot(d, d);
	float B = 2 * dot(d, (o - ce));
	float C = dot((o - ce), (o - ce)) - rc * rc;
	float root = B * B - 4 * A * C;
	
	if (root > 0) {
		//cout << -B - sqrt(root) << endl;
		float t1 = (-B - sqrt(root)) / (2 * A);
		float t2 = (-B + sqrt(root)) / (2 * A);
		
		//if (t1 == 0) t = t2;
		//if (t1 <= tmin) return false;
		if ((t1<=tmin && t2 <= tmin) || (t1>=tmax && t2>=tmax) || (t1<=tmin && t2 >= tmax)) return false;
		float t = t1;
		if (t <= 0.01f) t = 0;
		if (t <= tmin) t = t2;
		if (t <= 0.01f) t = 0;
		//if(t<= tmin) t = t2;
		if (t <= tmin || t >= tmax) { return false; }
		//cout << t << endl;
		rec.p = r.origin() + t * r.direction();
		vec3 v = rec.p - ce;
		v.make_unit_vector();
		rec.nv = v;
		rec.t = t;
		return true;
	}
	else return false;
}

vec3 reflect(const vec3 &d, const vec3 &nv) {
	/*
	To-do:
		compute the reflect direction
	*/
	vec3 tmp_d = d;
	vec3 tmp_n = nv;
	
	vec3 r = tmp_d - 2 * dot(tmp_d, tmp_n) * tmp_n;
	return r;
}

vec3 refract(const vec3& L, const vec3& N, float e) {
	/*
	To-do:
		compute the refracted(transmitted) direction
	*/
	vec3 tmpL = L;
	vec3 tmpN = N;
	tmpL.make_unit_vector();
	tmpN.make_unit_vector();
	float cosi = -dot(tmpL , tmpN);
	float e2sini2 = e*e*(1 - cosi * cosi);
	float cost2 = 1.0f - (e2sini2);
	//cout << cost2 << endl;
	if (cost2 <= 0) return reflect(tmpL, tmpN);
	vec3 t = e * tmpL + (e * cosi - sqrt(abs(cost2))) * tmpN;
	return t;
}
#endif