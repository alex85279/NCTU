#include <iostream>
#include <fstream>
#include <windows.h>
#include <GL/freeglut.h>

using namespace std;

GLbyte* pImage = NULL, * image = NULL;
GLint iWidth = 200, iHeight = 100, iComponents;
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


void init(GLbyte* pixels){
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

int main(int argc, char* argv[])
{
	int width = int(iWidth);
	int height = int(iHeight);
	GLbyte* data = new GLbyte[width * height * 3];

	fstream file;
	file.open("../ray.ppm", ios::out);

	file << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			float r = float(i) / float(width);
			float g = float(j) / float(height);
			float b = 0.2;

			file << int(r * 255) << " " << int(g * 255) << " " << int(b * 255) << "\n";

			int index = ((j) * width + i) * 3;

			data[index + 0] = (GLbyte)(r * 255);
			data[index + 1] = (GLbyte)(g * 255);
			data[index + 2] = (GLbyte)(b * 255);

		}
	}

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