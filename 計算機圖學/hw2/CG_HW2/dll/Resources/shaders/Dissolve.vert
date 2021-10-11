#version 330 core

//////////////////// TODO /////////////////////
/*
* Hint:
* 1. Receive position, normal, texcoord from bind buffer
* 2. Receive Model matrix, View matrix, and Projection matrix from uniform
* 3. Pass texcoord, worldPos and Normal to fragment shader
* 4. Calculate view space by gl_Position (must be vec4)
*/

layout(location = 0) in vec4 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec2 texCoord;
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
out vec2 uv;

out vec3 self_pos;
out vec3 normal;
void main() {
    gl_Position = P * V * M * pos;
    uv = texCoord;
	normal = norm;
	self_pos = vec3(pos.x,pos.y,pos.z);
}