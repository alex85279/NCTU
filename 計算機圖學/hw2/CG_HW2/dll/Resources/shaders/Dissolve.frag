#version 330 core

//////////////////// TODO /////////////////////
/*
* Hint:
* 1. Receive mainTex, WorldLightPos and WorldCamPos from uniform
* 2. Recieve all light color you defined in opengl from uniform
* 3. Recieve texcoord, worldPos and Normal from vertex shader
* 4. Calculate and return final color to opengl
*/
vec4 Ka = vec4(1, 1, 1, 1);
vec4 Kd = vec4(1, 1, 1, 1);
vec4 Ks = vec4(1, 1, 1, 1);
vec4 La = vec4(0.2, 0.2, 0.2, 1);
vec4 Ld = vec4(0.8, 0.8, 0.8, 1);
vec4 Ls = vec4(0.5, 0.5, 0.5, 1);
float gloss =100;
vec4 albedo;
float noise;
vec4 ambient;
vec4 diffuse;
vec4 specularPhong;
vec4 specularBlinn;
vec4 specular;
//in vec3 normal;
out vec4 outColor;
vec4 color;
uniform sampler2D mainTex;
uniform sampler2D noiseTex;

uniform float edge_length;
float thresh_hold;
float flag;

in vec2 uv;
in vec3 self_pos;
in vec3 normal;
void main() {
	thresh_hold = edge_length/2;
	albedo = texture2D(mainTex, uv);
	noise = texture(noiseTex,uv).x;
	if(noise - thresh_hold <0){
		discard;
	}
	flag = step(edge_length/2, thresh_hold + edge_length - noise);
	color = mix(albedo, vec4(1,0,0,1), flag);
    outColor = color;

}