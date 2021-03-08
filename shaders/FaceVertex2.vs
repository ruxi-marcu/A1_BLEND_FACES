#version 330 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;

out vec4 myvertex;  //Position
out vec3 mynormal;  //Normal

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
uniform vec3 LightPosW;


void main(){
	
  gl_Position =  proj * view * model * vec4(vertexPosition,1);
	
  mynormal = mat3(transpose(inverse(model))) * vertexNormal;
  myvertex = model * vec4(vertexPosition,1.0f);
	
}
