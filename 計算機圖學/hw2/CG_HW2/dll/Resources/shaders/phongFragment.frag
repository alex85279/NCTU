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
vec4 ambient;
vec4 diffuse;
vec4 specularPhong;
vec4 specularBlinn;
vec4 specular;
//in vec3 normal;
out vec4 outColor;

uniform sampler2D mainTex;

uniform vec3 WorldLightPos;
uniform vec3 WorldCamPos;

in vec2 uv;
in vec3 self_pos;
in vec3 normal;
void main() {
	vec3 L = normalize(WorldLightPos - self_pos);
	vec3 N = normalize(normal);
	vec3 V = normalize(WorldCamPos - self_pos);
	vec3 R = 2 * dot(L,N) * N - L;
	float disLV = sqrt(L.x * V.x + L.y * V.y + L.z * V.z);
	vec3 H = (L + V)/disLV;
	albedo = texture2D(mainTex, uv);
	ambient = La * Ka * albedo;
	diffuse = Ld * Kd * albedo * dot(L,N);
	specularPhong = Ls * Ks * pow(dot(V,R), gloss/4.0);
	specularBlinn = Ls * Ks * pow(dot(N,H), gloss);
	specular = mix(specularPhong, specularBlinn, 0);
    outColor = ambient + diffuse + specular;
}