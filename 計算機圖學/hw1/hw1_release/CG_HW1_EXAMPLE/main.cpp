#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <iostream>
#include <stb_image.h>
using namespace std;

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(void)
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(800, 600, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);


    // Generate textures
    unsigned int MyTexture;
    glGenTextures(1, &MyTexture);
    glBindTexture(GL_TEXTURE_2D, MyTexture);
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



    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float)height;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, MyTexture);
        

        glBegin(GL_QUADS);
        // Bottom left
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(-0.5f, 0.5f, 0.0f);

        // Top left
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(-0.5f, -0.5f, 0.0f);

        // Top right
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(0.5f, -0.5f, 0.0f);

        // Bottom right
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(0.5f, 0.5f, 0.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}