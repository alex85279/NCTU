#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include <windows.h>
#include <GL/freeglut.h>

#include "ray.h"
#include "vec3.h"
#include "geo.h"
#include "CImg.h"
#include <WinGDI.h>

float MAX(float a, float b) { return (a>b) ? a : b; }
using namespace std;
using namespace cimg_library;
int max_step = 5;

GLbyte* pImage = NULL, * image = NULL;
/* Change the width and height for the final result*/
GLint iWidth = 600, iHeight = 300, iComponents;
GLenum eFormat;

void RenderScene() {

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0);
	if (image != NULL) {
		if (pImage == NULL)
			pImage = image;
		glDrawPixels(iWidth, iHeight, GL_RGB, GL_UNSIGNED_BYTE, pImage);
	}
	glutSwapBuffers();
}


void init(GLbyte* pixels) {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	pImage = image = pixels;
}

void finish(void) {
	if (pImage != image) {
		free(pImage);
	}
	free(image);
}

void ChangeSize(int w, int h) {
	if (h == 0)
		h = 1;
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluOrtho2D(0.0f, (GLfloat)w, 0.0, (GLfloat)h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

vec3 shading(vec3 &lightsource, vec3 &intensity, hit_record ht, vec3 kd, const vector<sphere> &list) {
	/*
	To-do:
		define L, N by yourself
	*/
	vec3 L;
	vec3 N;
	L = lightsource - ht.p;
	L.make_unit_vector();
	N = ht.nv;
	ray shadowRay(ht.p, L);

	int intersect = -1;
	hit_record rec;
	float closest = FLT_MAX;
	/*
	To-do:
		To find whether the shadowRay hit other object,
		you should run the function "hit" of all the hitable you created
	*/
	for (int i = 0; i < list.size(); i++) {
		if (list[i].hit(shadowRay, 0, 500, rec)) {
			//cout << rec.t << endl;
			if (rec.t < closest) {
				closest = rec.t;
				intersect = i;

			}
		}
	}

	if (intersect == -1) {
		return kd*intensity*MAX(0, dot(N, unit_vector(L)));
	}
	else {
		//cout << dot(rec.nv, shadowRay.direction()) << endl;
		float shadow_factor = -dot(rec.nv, shadowRay.direction());
		if (shadow_factor <= 0) shadow_factor = 0;
		//cout << shadow_factor << endl;
		return kd * intensity * MAX(0, dot(N, unit_vector(L))) * (1-shadow_factor) ;
	}
}

vec3 skybox(const ray &r) {
	vec3 uni_direction = unit_vector(r.direction());
	float t = 0.5*(uni_direction.y() + 1);
	return (1.0 - t)* vec3(1, 1, 1) + t* vec3(0.5, 0.7, 1.0);
}

vec3 trace(const ray&r, const vector<sphere> &list, int depth) {
	if (depth >= max_step) return skybox(r); //or return vec3(0,0,0);

	int intersect = -1;
	hit_record rec;
	float closest = FLT_MAX;
	float min_t = 10000;
	for (int i = 0; i < list.size(); i++) {
		if (list[i].hit(r, 0, 500, rec)) {
			//cout << rec.t << endl;
			if (rec.t < min_t) {
				min_t = rec.t;
				intersect = i;
				
			}
		}
	}
	

	if (intersect != -1) {
		
		vec3 lightPosition = vec3(-10, 10, 0);
		vec3 lightIntensity = vec3(1, 1, 1);
		list[intersect].hit(r, 0, 500, rec);
		//diffuse
		vec3 diffuse_color = shading(lightPosition, lightIntensity, rec, list[intersect].kd, list);

		//reflect
		vec3 q = r.direction();
		vec3 n = rec.nv;
		q.make_unit_vector();
		n.make_unit_vector();
		vec3 reflect_param = reflect(q, n);
		reflect_param.make_unit_vector();
		ray reflectRay(rec.p, reflect_param);
		vec3 reflect_color = trace(reflectRay,list, depth + 1);
		//vec3 reflect_color = vec3(0,0,0);

		//refract
		vec3 L = r.direction();
		vec3 N = rec.nv;
		float e = 0.66f;
		L.make_unit_vector();
		N.make_unit_vector();
		vec3 refract_param = refract(L, N, e);
		refract_param.make_unit_vector();
		
		ray refractRay(rec.p, refract_param);
		vec3 refract_color;
		//test
		vec3 tmp_p = rec.p;
		if (list[intersect].hit(refractRay, 0, 500, rec) == true) {
			
			vec3 L2 = refractRay.direction();
			vec3 N2 = rec.nv * -1;
			float e2 = 1.55f;
			L2.make_unit_vector();
			N2.make_unit_vector();
			
			vec3 refract_param2 = refract(L2, N2, e2);
			//refract_param2.make_unit_vector();
			ray refractRay2(rec.p, refract_param2);
			refract_color = trace(refractRay2, list, depth + 1);
		}
		else {
			
			refract_color = trace(refractRay,list, depth+1);
		}
		//refract_color = trace(refractRay, list, depth + 1);
		float wr = list[intersect].w_r;
		float wt = list[intersect].w_t;
		
		vec3 color =  wt * refract_color + (1-wt) * (wr * reflect_color + (1 - wr) * diffuse_color);
		return color;	
	}
	else {
		return skybox(r);
	}
}
typedef struct BGR
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
}BGR;

typedef struct RGB
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
}RGB;


int main(int argc, char* argv[])
{ 
	// You should change the resolution when you hand in your result
	int width = int(iWidth);
	int height = int(iHeight);
	GLbyte* data = new GLbyte[width * height * 3];
	srand(time(NULL));

	//camera and projection plane
	vec3 lower_left_corner(-2, -1, -1);
	vec3 origin(0, 0, 0);
	vec3 horizontal(4, 0, 0);
	vec3 vertical(0, 2, 0);

	vec3 colorlist[8] = { vec3(0.8, 0.3, 0.3), vec3(0.3, 0.8, 0.3), vec3(0.3, 0.3, 0.8),
		vec3(0.8, 0.8, 0.3), vec3(0.3, 0.8, 0.8), vec3(0.8, 0.3, 0.8),
		vec3(0.8, 0.8, 0.8), vec3(0.3, 0.3, 0.3) };

	//test scene with spheres
	vector<sphere> hitable_list;
	hitable_list.push_back(sphere(vec3(0, -100.5, -2), 100)); //ground default color (1.0,1.0,1.0)
	hitable_list.push_back(sphere(vec3(0, 0, -2), 0.5, vec3(1.0f, 1.0f, 1.0f), 0.0f, 0.9f)); // refracted ball
	hitable_list.push_back(sphere(vec3(1, 0, -1.75), 0.5, vec3(1.0f, 1.0f, 1.0f), 0.9f, 0.0f)); // reflected ball
	hitable_list.push_back(sphere(vec3(-1, 0, -2.25), 0.5, vec3(1.0f, 0.7f, 0.3f), 0.0f, 0.0f)); // diffuse ball
	for (int i = 0; i < 48; i++) {
		float rand_max = 100;
		float xr = ((float)rand() / (float)(RAND_MAX)) * 6.0f - 3.0f;
		float zr = ((float)rand() / (float)(RAND_MAX)) * 3.0f - 1.5f;
		int cindex = rand() % 8;
		float rand_reflec = ((float)rand() / (float)(RAND_MAX));
		float rand_refrac = ((float)rand() / (float)(RAND_MAX));
		hitable_list.push_back(sphere(vec3(xr, -0.4, zr - 2), 0.1, colorlist[cindex], rand_reflec, 0.0f)); // small balls are all reflected ray.
		hitable_list.push_back(sphere(vec3(xr, -0.4, zr - 2), 0.1, colorlist[cindex], 0.0f, rand_refrac)); // small balls are all refracted ray.
		hitable_list.push_back(sphere(vec3(xr, -0.4, zr - 2), 0.1, colorlist[cindex], 0.0f, 0.0f)); // small balls are neither reflected nor refracted.
	}



	/*
		To-do:
			Save the result to ppm and bmp/jpg/png format
	*/
	fstream file;
	file.open("../ray.ppm", ios::out);
	file << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			float u = float(i) / float(width);
			float v = float(j) / float(height);
			ray r(origin, lower_left_corner + u*horizontal + v*vertical);
			vec3 c = trace(r, hitable_list, 0);

			/*Hint: Here to save each pixel after ray tracing*/
		
			

			file << int(c.r() * 255) << " " << int(c.g() * 255) << " " << int(c.b() * 255) << "\n";

			// for display window
			int index = ((j) * width + i) * 3;
			data[index + 0] = (GLbyte)(c.r() * 255);
			data[index + 1] = (GLbyte)(c.g() * 255);
			data[index + 2] = (GLbyte)(c.b() * 255);
		}
	}
	
	file.close();
	
	CImg<unsigned char> im("../ray.ppm");
	im.save("../out.bmp");

	
	///////////////////
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GL_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Image Loading Test");

	glutReshapeFunc(ChangeSize);
	glutDisplayFunc(RenderScene);
	init(data);
	glutMainLoop();
	finish();

	



	return 0;
}