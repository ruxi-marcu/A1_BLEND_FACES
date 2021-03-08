#version 330 core

in vec4 myvertex;  //Position
in vec3 mynormal;  //Normal

out vec4 fragColor;

uniform mat4 model;
uniform mat4 proj;
uniform mat4 view;
uniform vec3 LightPosW;

vec4 ambient = vec4(0.0, 0.0, 0.5, 1); //ka
vec4 diffuse = vec4(0.5, 0.5, 0.5, 1);  //kd
vec4 specular = vec4(1.0, 1.0, 1.0, 1);  //ks
float shininess = 100;

void main(){

	vec4 lightcolor = vec4(1,1,1,1);
	
	const vec3 eyepos = vec3(0,0,7) ;

	vec3 mypos = myvertex.xyz / myvertex.w ; 
    vec3 eyedirn = normalize(eyepos - mypos) ; 

    vec3 normal = normalize(mynormal) ; 
		 
	vec3 position = LightPosW;
	vec3 direction = normalize(position - mypos); 
    vec3 halfvec = normalize (direction + eyedirn) ; 

    float nDotL = dot(normal, direction)  ;    
    vec4 lambert = diffuse * lightcolor * max (nDotL, 0.0) ;  

    float nDotH = dot(normal, halfvec) ;
    vec4 phong = specular  * lightcolor * pow (max(nDotH, 0.0), shininess) ; 

	
	fragColor = (ambient + lambert + phong ) * vec4(1.0,1.0,1.0,1.0); 	
		

}