/*

CG Homework2 - Phong Shading, Dissolving effects and Ramp effects

Objective - learning GLSL, glm model datatype(for .obj file)

Overview:

	1. Render the model with texture using VBO and/or VAO

	2. Implement Phong shading

	3. Implement Dissolving effects via shader

	4. Implement Ramp effects via shader (Basically a dicrete shading)

Good luck!

*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h> /*for function: offsetof */
#include <math.h>
#include <string.h>

#include "GL/glew.h"
#include "GL/freeglut.h"
#include "glm/glm.h"
#include "glmMathematics/glm/gtc/matrix_transform.hpp"
#include "glmMathematics/glm/glm.hpp"

#include <vector>
#include "Model.hpp"
#include "Shader.hpp"
#include "Input.hpp"
GLuint TextureID, normalTextureID, noiseTextureID, rampTextureID;
GLfloat normalWid, normalHei;
GLuint FramebufferName = 0;
GLuint renderedTexture;

glm::vec3 LightCenter = glm::vec3(0, 15, 6);

int g_width = 512;
int g_height = 512;

void mykeyboard(unsigned char key, int x, int y);
float edgeLength = 0.5;


void InitShader()
{
	///////////////// TODO ///////////////////
	/*
	* Hint:
	* 1. ReadShader("shader path")
	* 2. CreateShader()
	* 3. CreateProgram(ExampleProgram)
	*/
	char* ph_vertexShaderSource = ReadShader("Resources/shaders/phongFragment.vert");
	char* ph_fragmentShaderSource = ReadShader("Resources/shaders/phongFragment.frag");
	CreateShader(vertexShader, GL_VERTEX_SHADER, ph_vertexShaderSource);
	CreateShader(fragmentShader, GL_FRAGMENT_SHADER, ph_fragmentShaderSource);
	CreateProgram(PhongProgram, 2, vertexShader, fragmentShader);

	char* ds_vertexShaderSource = ReadShader("Resources/shaders/Dissolve.vert");
	char* ds_fragmentShaderSource = ReadShader("Resources/shaders/Dissolve.frag");
	CreateShader(vertexShader, GL_VERTEX_SHADER, ds_vertexShaderSource);
	CreateShader(fragmentShader, GL_FRAGMENT_SHADER, ds_fragmentShaderSource);
	CreateProgram(DissolveProgram, 2, vertexShader, fragmentShader);


	char* toon_vertexShaderSource = ReadShader("Resources/shaders/Toon.vert");
	char* toon_fragmentShaderSource = ReadShader("Resources/shaders/Toon.frag");
	CreateShader(vertexShader, GL_VERTEX_SHADER, toon_vertexShaderSource);
	CreateShader(fragmentShader, GL_FRAGMENT_SHADER, toon_fragmentShaderSource);
	CreateProgram(ToonProgram, 2, vertexShader, fragmentShader);

	char* rt_vertexShaderSource = ReadShader("Resources/shaders/real_Toon.vert");
	char* rt_fragmentShaderSource = ReadShader("Resources/shaders/real_Toon.frag");
	CreateShader(vertexShader, GL_VERTEX_SHADER, rt_vertexShaderSource);
	CreateShader(fragmentShader, GL_FRAGMENT_SHADER, rt_fragmentShaderSource);
	CreateProgram(rtProgram, 2, vertexShader, fragmentShader);




}
GLuint loadTexture(char* name, GLfloat width, GLfloat height)
{
	return glmLoadTexture(name, false, true, true, true, &width, &height);
}
void InitTexture()
{
	TextureID = loadTexture(main_tex_dir, 512, 256);
	noiseTextureID = loadTexture(noise_tex_dir, 360, 360);
	rampTextureID = loadTexture(ramp_tex_dir, 256, 256);
}
void BindBuffer()
{
	///////////////// TODO ///////////////////
	/*
	* Hint:
	* 1. Setup VAO
	* 2. Setup VBO of vertex positions, normals, and texcoords
	*/

	int position_size = positions.size();
	int tri_num = model->numtriangles;

	
	//std::cout << sizeof(bunny_vertices) << std::endl;
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexAttribute) * 196890, b_vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttribute), (void*)(offsetof(VertexAttribute, position)));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttribute), (void*)(offsetof(VertexAttribute, normal)));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexAttribute), (void*)(offsetof(VertexAttribute, texcoord)));
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);





}
void InitBuffer()
{
	///////////////// TODO ///////////////////
	/*
	* Hint:
	1. Generate vertex array object and buffer object names
	2. Bind buffers
	*/
	glGenVertexArrays(1, &VAO);
	glGenBuffers(3, VBO);

	BindBuffer();



}
void init(void) {
	glEnable(GL_CULL_FACE);
	InitTexture();
	LoadObj(obj_file_dir);
	InitShader();
	InitBuffer();
}
glm::mat4 getV()
{
	// set camera position and configuration
	glm::vec3 cameraPos = glm::vec3(eyex, eyey, eyez);
	glm::vec3 cameraFront = glm::vec3(cos(eyet * M_PI / 180) * cos(eyep * M_PI / 180), sin(eyet * M_PI / 180), -1.0 * cos(eyet * M_PI / 180) * sin(eyep * M_PI / 180));
	glm::vec3 cameraRight = glm::vec3(0.0, 1.0, 0.0);
	return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraRight);
}
glm::mat4 getP()
{
	// set perspective view
	float fov = 45.0f;
	float aspect = g_width / g_height; // since window is (512, 512)
	float nearDistance = 0.001f;
	float farDistance = 1000.0f;
	return glm::perspective(glm::radians(fov), aspect, nearDistance, farDistance);
}
void RenderObject(int mode)
{
	GLuint currentProgram;
	if (mode == 0 || mode == 1) {
		// Phong
		currentProgram = PhongProgram;
	}
	else if (mode == 2)
	{
		currentProgram = DissolveProgram;
	}
	else if (mode == 3)
	{
		currentProgram = ToonProgram;
	}
	else if (mode == 4)
	{
		currentProgram = rtProgram;
	}
	glUseProgram(currentProgram);
	
	
	glm::mat4 M(1.0f);
	M = glm::translate(M, glm::vec3(ball_pos[0], ball_pos[1], ball_pos[2]));
	M = glm::rotate(M, ball_rot[0], glm::vec3(1, 0, 0));
	M = glm::rotate(M, ball_rot[1], glm::vec3(0, 1, 0));
	M = glm::rotate(M, ball_rot[2], glm::vec3(0, 0, 1));
		
	GLuint ModelMatrixID = glGetUniformLocation(currentProgram, "M");
	glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &M[0][0]);
	
	///////////////// TODO ///////////////////
	/*
	* Hint:
	* 1. Also pass projection matrix, and view matrix and trigger by Uniform (getP() and getV())
	*/
	
	

	
	





	// pass main textures to shader and trigger by Uniform
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, TextureID);
	glUniform1i(glGetUniformLocation(currentProgram, "mainTex"), 0);
	
	if (mode == 0 || mode == 1) {
		// Phong
		GLuint WorldLightPosID = glGetUniformLocation(currentProgram, "WorldLightPos");
		GLuint WorldCamID = glGetUniformLocation(currentProgram, "WorldCamPos");

		glUniform3f(WorldCamID, eyex, eyey, eyez);
		glUniform3f(WorldLightPosID, light_pos[0], light_pos[1], light_pos[2]);

		///////////////// TODO ///////////////////
		/*
		* Hint:
		* 1. Set up light color
		* 2. Pass all variable to shaders and trigger by Uniform
		*/
		

		glm::mat4 V = getV();
		GLfloat Vmtx[16];
		for (int i = 0; i < 4; i++) {
			Vmtx[i * 4 + 0] = V[i].x;
			Vmtx[i * 4 + 1] = V[i].y;
			Vmtx[i * 4 + 2] = V[i].z;
			Vmtx[i * 4 + 3] = V[i].w;

		}
		GLint VLoc = glGetUniformLocation(currentProgram, "V");
		glUniformMatrix4fv(VLoc, 1, GL_FALSE, Vmtx);
		glm::mat4 P = getP();
		GLfloat Pmtx[16];
		for (int i = 0; i < 4; i++) {
			Pmtx[i * 4 + 0] = P[i].x;
			Pmtx[i * 4 + 1] = P[i].y;
			Pmtx[i * 4 + 2] = P[i].z;
			Pmtx[i * 4 + 3] = P[i].w;
		}
		GLint PLoc = glGetUniformLocation(currentProgram, "P");
		glUniformMatrix4fv(PLoc, 1, GL_FALSE, Pmtx);

		
		
		
		



	}
	else if (mode == 2)
	{
		// Dissolve
		///////////////// TODO ///////////////////
		/*
		* Hint:
		* 1. Pass noise texture to shader and trigger by Uniform
		* 2. Set up edge color and width when dissoving
		* 3. Pass edge color, width and dissolveThreshold to shaders and trigger by Uniform
		*/
		glm::mat4 V = getV();
		GLfloat Vmtx[16];
		for (int i = 0; i < 4; i++) {
			Vmtx[i * 4 + 0] = V[i].x;
			Vmtx[i * 4 + 1] = V[i].y;
			Vmtx[i * 4 + 2] = V[i].z;
			Vmtx[i * 4 + 3] = V[i].w;

		}
		GLint VLoc = glGetUniformLocation(currentProgram, "V");
		glUniformMatrix4fv(VLoc, 1, GL_FALSE, Vmtx);
		glm::mat4 P = getP();
		GLfloat Pmtx[16];
		for (int i = 0; i < 4; i++) {
			Pmtx[i * 4 + 0] = P[i].x;
			Pmtx[i * 4 + 1] = P[i].y;
			Pmtx[i * 4 + 2] = P[i].z;
			Pmtx[i * 4 + 3] = P[i].w;
		}
		GLint PLoc = glGetUniformLocation(currentProgram, "P");
		glUniformMatrix4fv(PLoc, 1, GL_FALSE, Pmtx);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, noiseTextureID);
		glUniform1i(glGetUniformLocation(currentProgram, "noiseTex"), 1);

		GLint EdgeL = glGetUniformLocation(currentProgram, "edge_length");
		glUniform1f(EdgeL, edgeLength);




	}
	else if (mode == 3)
	{
		// Outline + Ramp
		GLuint WorldLightPosID = glGetUniformLocation(currentProgram, "WorldLightPos");
		glUniform3f(WorldLightPosID, light_pos[0], light_pos[1], light_pos[2]);

		///////////////// TODO ///////////////////
		/*
		* Hint:
		* 1. Pass ramp texture to shader and trigger by Uniform
		*/
		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, rampTextureID);
		glUniform1i(glGetUniformLocation(currentProgram, "rampTex"), 2);
		glm::mat4 V = getV();
		GLfloat Vmtx[16];
		for (int i = 0; i < 4; i++) {
			Vmtx[i * 4 + 0] = V[i].x;
			Vmtx[i * 4 + 1] = V[i].y;
			Vmtx[i * 4 + 2] = V[i].z;
			Vmtx[i * 4 + 3] = V[i].w;

		}
		GLint VLoc = glGetUniformLocation(currentProgram, "V");
		glUniformMatrix4fv(VLoc, 1, GL_FALSE, Vmtx);
		glm::mat4 P = getP();
		GLfloat Pmtx[16];
		for (int i = 0; i < 4; i++) {
			Pmtx[i * 4 + 0] = P[i].x;
			Pmtx[i * 4 + 1] = P[i].y;
			Pmtx[i * 4 + 2] = P[i].z;
			Pmtx[i * 4 + 3] = P[i].w;
		}
		GLint PLoc = glGetUniformLocation(currentProgram, "P");
		glUniformMatrix4fv(PLoc, 1, GL_FALSE, Pmtx);




	}
	else if (mode == 4)
	{
	// Outline + Ramp
	GLuint WorldLightPosID = glGetUniformLocation(currentProgram, "WorldLightPos");
	glUniform3f(WorldLightPosID, light_pos[0], light_pos[1], light_pos[2]);

	///////////////// TODO ///////////////////
	/*
	* Hint:
	* 1. Pass ramp texture to shader and trigger by Uniform
	*/
	glActiveTexture(GL_TEXTURE0 + 2);
	glBindTexture(GL_TEXTURE_2D, rampTextureID);
	glUniform1i(glGetUniformLocation(currentProgram, "rampTex"), 2);
	glm::mat4 V = getV();
	GLfloat Vmtx[16];
	for (int i = 0; i < 4; i++) {
		Vmtx[i * 4 + 0] = V[i].x;
		Vmtx[i * 4 + 1] = V[i].y;
		Vmtx[i * 4 + 2] = V[i].z;
		Vmtx[i * 4 + 3] = V[i].w;

	}
	GLint VLoc = glGetUniformLocation(currentProgram, "V");
	glUniformMatrix4fv(VLoc, 1, GL_FALSE, Vmtx);
	glm::mat4 P = getP();
	GLfloat Pmtx[16];
	for (int i = 0; i < 4; i++) {
		Pmtx[i * 4 + 0] = P[i].x;
		Pmtx[i * 4 + 1] = P[i].y;
		Pmtx[i * 4 + 2] = P[i].z;
		Pmtx[i * 4 + 3] = P[i].w;
	}
	GLint PLoc = glGetUniformLocation(currentProgram, "P");
	glUniformMatrix4fv(PLoc, 1, GL_FALSE, Pmtx);




	}
	
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, model->numtriangles * 3);
	glBindVertexArray(0);
	
}
void display(void)
{
	BindBuffer();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	RenderObject(mode + 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glUseProgram(0);
	gluLookAt(eyex, eyey, eyez, eyex + cos(eyet * M_PI / 180) * cos(eyep * M_PI / 180), eyey + sin(eyet * M_PI / 180), eyez - cos(eyet * M_PI / 180) * sin(eyep * M_PI / 180), 0.0, 1.0, 0.0);
	// Render Light Bulb
	draw_light_bulb();
	// Update Inputs
	camera_light_ball_move();
	glutSwapBuffers();
}
//no need to modify the following functions
void reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.001f, 1000.0f);
	g_width = width;
	g_height = height;
	glMatrixMode(GL_MODELVIEW);
}
void idle(void)
{
	glutPostRedisplay();
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("CG_HW2_309553012");
	glutReshapeWindow(g_width, g_height);
	glewInit();
	init();
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutKeyboardFunc(mykeyboard);
	glutKeyboardUpFunc(keyboardup);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMainLoop();
	return 0;
}

void mykeyboard(unsigned char key, int x, int y) {
	switch (key) {
		case 'b'://toggle bump mapping
		{
			mode = (mode + 1) % 4;
			printf("Now Mode = %d\n", mode);
			break;
		}
		
		///////////////// TODO ///////////////////
		/* Hint: (Optional)
		* 1. Press keyboard to change dissolve threshold (or use mouse event or run automatically)
		*/

		case '+':
		{
			edgeLength += 0.1;
			if (edgeLength >= 2) {
				edgeLength = 2;
			}
			break;
		}
		case '-':
		{
			edgeLength -= 0.1;
			if (edgeLength <= 0) {
				edgeLength = 0;
			}
			break;
		}





		// other key event
		case 27:
		{	//ESC
			break;
		}
		case 'd':
		{
			right = true;
			break;
		}
		case 'a':
		{
			left = true;
			break;
		}
		case 'w':
		{
			forward = true;
			break;
		}
		case 's':
		{
			backward = true;
			break;
		}
		case 'q':
		{
			up = true;
			break;
		}
		case 'e':
		{
			down = true;
			break;
		}
		case 't':
		{
			lforward = true;
			break;
		}
		case 'g':
		{
			lbackward = true;
			break;
		}
		case 'h':
		{
			lright = true;
			break;
		}
		case 'f':
		{
			lleft = true;
			break;
		}
		case 'r':
		{
			lup = true;
			break;
		}
		case 'y':
		{
			ldown = true;
			break;
		}
		case 'i':
		{
			bforward = true;
			break;
		}
		case 'k':
		{
			bbackward = true;
			break;
		}
		case 'l':
		{
			bright = true;
			break;
		}
		case 'j':
		{
			bleft = true;
			break;
		}
		case 'u':
		{
			bup = true;
			break;
		}
		case 'o':
		{
			bdown = true;
			break;
		}
		case '7':
		{
			bx = true;
			break;
		}
		case '8':
		{
			by = true;
			break;
		}
		case '9':
		{
			bz = true;
			break;
		}
		case '4':
		{
			brx = true;
			break;
		}
		case '5':
		{
			bry = true;
			break;
		}
		case '6':
		{
			brz = true;
			break;
		}

		//special function key
		case 'z'://move light source to front of camera
		{
			light_pos[0] = eyex + cos(eyet * M_PI / 180) * cos(eyep * M_PI / 180);
			light_pos[1] = eyey + sin(eyet * M_PI / 180);
			light_pos[2] = eyez - cos(eyet * M_PI / 180) * sin(eyep * M_PI / 180);
			break;
		}
		case 'x'://move ball to front of camera
		{
			ball_pos[0] = eyex + cos(eyet * M_PI / 180) * cos(eyep * M_PI / 180) * 3;
			ball_pos[1] = eyey + sin(eyet * M_PI / 180) * 5;
			ball_pos[2] = eyez - cos(eyet * M_PI / 180) * sin(eyep * M_PI / 180) * 3;
			break;
		}
		case 'c'://reset all pose
		{
			//1.100000 1.500000 1.300000
			light_pos[0] = 1.1;
			light_pos[1] = 1.5;
			light_pos[2] = 1.3;
			ball_pos[0] = 0;
			ball_pos[1] = 0;
			ball_pos[2] = 0;
			ball_rot[0] = 0;
			ball_rot[1] = 0;
			ball_rot[2] = 0;
			eyex = -0.3;
			eyey = 1.1;
			eyez = 6.6;
			eyet = 0;
			eyep = 90;
			dissolveThreshold = 0;
			break;
		}
		default:
		{
			break;
		}
	}
}
