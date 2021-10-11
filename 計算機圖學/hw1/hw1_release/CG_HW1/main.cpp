#include <GLFW/glfw3.h>
#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <stb_image.h>

#define M_PI 3.14159265358979323846

using namespace std;
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

static void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);


const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

float angle;
int render_mode = 0;
float move_x = 0;
float move_y = 0;
float move_z = 0;
float move_step = 0.05f;
int left_key_pressed = 0;
double x_position;
double pre_x;
double y_position;
double pre_y;
double animation_flag = 0;
double animation_timer = 0;
double now_time;
double pre_time;
//------------- TO DO ------------- 
// Declaration: ex. mouse events variables, camera positions ...

glm::vec3 ball_cal(float u, float v, float r) {
    GLfloat x = cos(u) * sin(v) * r;
    GLfloat y = cos(v) * r;
    GLfloat z = sin(u) * sin(v) * r;
    return glm::vec3(x, y, z);
}
glm::vec3 normal_cal(float u, float v) {
    GLfloat x = cos(u) * sin(v);
    GLfloat y = cos(v);
    GLfloat z = sin(u) * sin(v);
    return glm::vec3(x, y, z);
}

void mySphere(double r, int slice, int stack) {
    //------------- TO DO ------------- 
    // Draw your sphere
    float Ustart = 0;
    float Uend = 2 * M_PI;
    float Vstart = 0;
    float Vend = M_PI;
    int point_num_U = slice;
    int point_num_V = stack;
    
    float stepU = (Uend - Ustart) / point_num_U;
    float stepV = (Vend - Vstart) / point_num_V;
    glBegin(GL_QUADS);
    for (int i = 0; i < point_num_U; i++) {
        
        for (int j = 0; j < point_num_V; j++) {
            float u = i * stepU + Ustart;
            float v = j * stepV + Vstart;
            float un = (i + 1 == point_num_U) ? Uend : (i + 1) * stepU + Ustart;
            float vn = (j + 1 == point_num_V) ? Vend : (j + 1) * stepV + Vstart;

            glm::vec3 p0, p1, p2, p3, n0, n1, n2, n3;
            double new_r = r;
            if (animation_flag == 1 && i % 10 == 0) {
                new_r = r + 0.5 * sin(animation_timer);
            }
            p0 = ball_cal(u, v, new_r);
            p1 = ball_cal(u, vn, new_r);
            p2 = ball_cal(un, v, new_r);
            p3 = ball_cal(un, vn, new_r);

            n0 = normal_cal(u, v);
            n1 = normal_cal(u, vn);
            n2 = normal_cal(un, v);
            n3 = normal_cal(un, vn);

            
            

            glNormal3f(n0.x, n0.y, n0.z);
            //glNormal3f(cos(u) * sin(v), cos(v), sin(u) * sin(v));
            glTexCoord2f(un / (2 * M_PI), v / (M_PI));
            glVertex3f(p0.x, p0.y, p0.z);

            glNormal3f(n1.x, n1.y, n1.z);
            //glNormal3f(cos(u) * sin(vn), cos(vn), sin(u) * sin(vn));
            glTexCoord2f(un / (2 * M_PI), v / (M_PI));
            glVertex3f(p1.x, p1.y, p1.z);

            glNormal3f(n3.x, n3.y, n3.z);
            //glNormal3f(cos(un) * sin(vn), cos(vn), sin(un) * sin(vn));
            glTexCoord2f(un / (2 * M_PI), v / (M_PI));
            glVertex3f(p3.x, p3.y, p3.z);

            glNormal3f(n2.x, n2.y, n2.z);
            //glNormal3f(cos(un) * sin(v), cos(v), sin(un) * sin(v));
            glTexCoord2f(un / (2 * M_PI), v / (M_PI));
            glVertex3f(p2.x, p2.y, p2.z);

            
            
        }
    }
    glEnd();




}







void myCube() {
    //------------- TO DO ------------- 
    // Draw your cube
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(2);
    //glNormal3f(n0.x, n0.y, n0.z);
    glm::vec3 n0, v1, v2;
    glBegin(GL_QUADS);
    //first face

    //cout <<"first face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(-0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    n0 = glm::vec3(-0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(-0.5f, -0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(-0.5f, -0.5f, 0.5f);

    //second face


    //cout << "2 face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(-0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    n0 = glm::vec3(-0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    //third face
    //cout << "3 face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    n0 = glm::vec3(0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    n0 = glm::vec3(0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    //forth face
    //cout << "4 face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    n0 = glm::vec3(0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(-0.5f, -0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(-0.5f, -0.5f, 0.5f);

    //fifth face
    //cout << "5 face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(-0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    n0 = glm::vec3(0.5f, 0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    n0 = glm::vec3(0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, 0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(-0.5f, -0.5f, 0.5f);

    //sixth face
    //cout << "6 face: " << n0.x << " " << n0.y << " " << n0.z << endl;
    n0 = glm::vec3(0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, 0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    n0 = glm::vec3(-0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(-0.5f, -0.5f, -0.5f);

    n0 = glm::vec3(0.5f, -0.5f, -0.5f);
    n0 = glm::normalize(n0);
    glNormal3f(n0.x, n0.y, n0.z);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    /*glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);*/
    glEnd();





}

int main(int argc, char** argv)
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CG_HW1_TA", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    unsigned int texture1, texture2, texture3;

    //------------- TO DO ------------- 
    // Generate textures
    // Generate textures
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    // load image, create texture and generate mipmaps
    int w, h, nrChannels;
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned char* data = stbi_load("../resources/container.jpg", &w, &h, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    }
    else
    {
        cout << "Failed to load texture" << endl;
    }
    stbi_image_free(data);

    //tex earth
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // load image, create texture and generate mipmaps
    
    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    data = stbi_load("../resources/earth.jpg", &w, &h, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    }
    else
    {
        cout << "Failed to load texture" << endl;
    }
    stbi_image_free(data);

    //tex moon
    glGenTextures(1, &texture3);
    glBindTexture(GL_TEXTURE_2D, texture3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // load image, create texture and generate mipmaps

    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    data = stbi_load("../resources/moon.jpg", &w, &h, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    }
    else
    {
        cout << "Failed to load texture" << endl;
    }
    stbi_image_free(data);





    //------------- TO DO ------------- 
    // Enable lighting
    GLfloat light_position[] = { 10,10,10, 1};
    GLfloat test_position[] = { -10,-10,-10, 1 };
    GLfloat ambient_color[] = { 0.5f, 0.5f, 0.5f, 1 };
    GLfloat diffuse_color[] = { 1, 1, 1, 1 };
    GLfloat specular_color[] = { 1, 1, 1, 1 };
   
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    
    
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_color);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_color);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular_color);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    
    
    






    angle = 0;
    pre_time = glfwGetTime();
    now_time = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        animation_timer = animation_timer + 1;
        if (animation_timer >= 360) animation_timer = 0;
        if (left_key_pressed == 1) {
            glfwGetCursorPos(window, &x_position, &y_position);
            //cout << "( " << x_position - pre_x << ", " << y_position - pre_y << ")" << endl;
            move_x += (x_position - pre_x) * 0.01f;
            move_y -= (y_position - pre_y) * 0.01f;
            pre_x = x_position;
            pre_y = y_position;
        }


        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        //ModelView Matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        gluLookAt(0 + move_x, 0 + move_y, 5 + move_z, 0 + move_x, 0 + move_y, -1 + move_z, 0, 1, 0);
        //Projection Matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        


        
        gluPerspective(45.0f, width / (GLfloat)height, 0.1, 1000);

        //Viewport Matrix
        glViewport(0, 0, width, height);

       

        //------------- TO DO ------------- 
        // Enable GL capabilities
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);
        //glEnable(GL_CULL_FACE);
        //glEnable(GL_NORMALIZE);
        //glEnable(GL_CULL_FACE);
        
        //glFrontFace(GL_CW);
        
        

        // clear
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //------------- TO DO ------------- 
        // Rotate, Draw and Switch objects
        now_time = glfwGetTime();

        angle = angle + 70  * (now_time - pre_time);
        if (angle >= 360) angle = 0;
        pre_time = now_time;
        
        
        glPushMatrix();
        
        glRotatef(angle, 1, 1, 0);
        if (render_mode == 0) {
            glBindTexture(GL_TEXTURE_2D, texture1);
            myCube();
        }
        else if (render_mode == 1) {
            glBindTexture(GL_TEXTURE_2D, texture2);
            mySphere(1, 360, 180);
        }
        else if (render_mode == 2) {
            glBindTexture(GL_TEXTURE_2D, texture3);
            mySphere(1, 360, 180);
            
        }
        
        glPopMatrix();
        
        
        


        glfwSwapBuffers(window);
        glfwPollEvents();


    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);

    return 0;
}

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    //------------- TO DO ------------- 
    // Define your keyboard event
    if (key == GLFW_KEY_1) {
        render_mode = 0;
    }
    if (key == GLFW_KEY_2) {
        render_mode = 1;
    }
    if (key == GLFW_KEY_3) {
        render_mode = 2;
    }
    if (key == GLFW_KEY_W) {
        move_y += move_step;
    }
    if (key == GLFW_KEY_S) {
        move_y -= move_step;
    }
    if (key == GLFW_KEY_D) {
        move_x += move_step;
    }
    if (key == GLFW_KEY_A) {
        move_x -= move_step;
    }
    if (key == GLFW_KEY_Q) {
        move_z += move_step;
    }
    if (key == GLFW_KEY_E) {
        move_z -= move_step;
    }
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        if (animation_flag == 0) {
            animation_flag = 1;
            animation_timer = 0;
        }
        else {
            animation_flag = 0;
            animation_timer = 0;
        }
    }
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    //------------- TO DO ------------- 
    // Define your mouse event
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            cout << "left pressed" << endl;
            left_key_pressed = 1;
            glfwGetCursorPos(window, &x_position, &y_position);
            pre_x = x_position;
            pre_y = y_position;
            //cout << "( " << x_position << ", " << y_position << ")" << endl;
        }
        if (action == GLFW_RELEASE) {
            cout << "left released" << endl;
            left_key_pressed = 0;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            cout << "right pressed" << endl;
            move_x = 0;
            move_y = 0;
            move_z = 0;
        }
        if (action == GLFW_RELEASE) {
            cout << "right released" << endl;
        }
    }

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    //------------- TO DO ------------- 
    // (optional) Define your scroll event
    //cout << yoffset << endl;
    move_z -= yoffset * 0.05f;
}