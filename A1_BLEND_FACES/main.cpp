
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
//#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <set>
#include <algorithm>

#include "shader_m.h"
#include "LoadObj.h"
#include "FilesInDir.h"
#include "vboindexer.h"

//#include "camera.h"   

#include <Eigen/Dense>
using namespace Eigen;


#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//window
int wWidth = 2560;
int wHeight = 1440;

//imGUI
static float f[25] = {0.0f};
float g = 0.0f;

//face
glm::mat4 modelTeapot;
glm::mat4 viewTeapot;
glm::mat4 projTeapot;

//Eigen 
int nrFaces = 25;
int nrIndices;
int nrxyzVertices;

//animation
bool runAnimation = false;
MatrixXf animMatrix;
int frame = 0;
static int nrFrames = 0;
std::vector<float> tmpAnimation;

//mouse ....
double xmouse, ymouse;
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
int tindex = -1;

//vertex picking
glm::vec3 ray;
std::vector<unsigned short> indices[25];
std::vector<glm::vec3> indexed_vertices[25];
std::vector<glm::vec2> indexed_uvs[25];
std::vector<glm::vec3> indexed_normals[25];
std::vector<glm::vec3> indexed_tangents[25];
std::vector<glm::vec3> indexed_bitangents[25];
std::vector<glm::vec3> allIntersects;

bool inters = false;




static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

bool RayIntersectsTriangle(glm::vec3 rayOrigin,
	glm::vec3 rayVector,
	glm::vec3 Triv0,
	glm::vec3 Triv1,
	glm::vec3 Triv2,
	std::vector<glm::vec3> intersectedTriangles,
	glm::vec3 &outIntersectionPoint,
	std::vector<float>  &tVector
)
{
	const float EPSILON = 0.0000001;
	glm::vec3 vertex0 = Triv0;
	glm::vec3 vertex1 = Triv1;
	glm::vec3 vertex2 = Triv2;
	glm::vec3 edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = cross(rayVector,edge2);
	a = dot(edge1,h);
	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.
	f = 1.0 / a;
	s = rayOrigin - vertex0;
	u = f * dot(s,h);
	if (u < 0.0 || u > 1.0)
		return false;
	q = cross(s,edge1);
	v = f * dot(rayVector,q);
	if (v < 0.0 || u + v > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * dot(edge2,q);
	if (t > EPSILON) // ray intersection
	{
		outIntersectionPoint = rayOrigin + rayVector * t;
		allIntersects.push_back(outIntersectionPoint);
		intersectedTriangles.push_back(Triv0);
		intersectedTriangles.push_back(Triv1);
		intersectedTriangles.push_back(Triv2);
		tVector.push_back(t);
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

VectorXf blendFaces( MatrixXf facesXf, VectorXf weightsXf) {

	for (int i = 1; i < nrFaces; i++)
		for (int j = 0; j < nrxyzVertices; j++)
			facesXf(i, j) = facesXf(i, j) - facesXf(0, j);

	VectorXf resXf(nrxyzVertices);
	resXf = facesXf.transpose() * weightsXf;

	return resXf;
}


int readAnimation(const char * path, std::vector<float> &values) {
	
	values.clear();
	std::ifstream f("animation.txt");

	if (!f.good())
	{
		printf("Error while opening the file");
		f.close();
		return 0;
	}

	while (!f.eof())
	{
		float val;
		f >> val;
		values.push_back(val);
	}
	values.pop_back();
	f.close();

	return 0;

}

MatrixXf setAnimationWeights(std::vector<float> readNum) {
	readNum.clear();
	int rowCount = nrFrames;
	int colCount = nrFaces - 1;
	MatrixXf animMatrixXf(rowCount, colCount);
	for(int i=0; i< rowCount; i++)
		for (int j=0; j< colCount; j++)
			animMatrixXf(i, j) = readNum[(i*colCount) +j];
		
	return animMatrixXf;
}

void resetAll() {
	f[0] = 1;
	for (int i = 0; i < nrFaces; i++)
		f[i] = 0;
	frame = 0;
	runAnimation = false;
}

glm::vec3 mousePicking(glm::mat4 projMatrix, glm::mat4 viewMatrix) {
	float x = (2.0f * xmouse) / wWidth - 1.0f;
	float y = 1.0f - (2.0f * ymouse) / wHeight;
	float z = 1.0f;
	glm::vec3 rayNDS = glm::vec3(x, y, z);

	glm::vec4 rayCLIP = glm::vec4(rayNDS.x, rayNDS.y, -1.0, 1.0);

	glm::vec4 rayEYE = inverse(projMatrix) * rayCLIP;
	rayEYE = glm::vec4(rayEYE.x, rayEYE.y, -1.0, 0.0);

	glm::vec3 rayWORLD = glm::vec3(inverse(viewMatrix) * rayEYE);
	rayWORLD = glm::normalize(rayWORLD);

	return  rayWORLD;

}

//-----------------------------------------------------------------------------------------------------------MAIN
int main(int, char**)
{
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;


	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only


	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(wWidth, wHeight, "Face Animation", NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	//glfwSetCursorPosCallback(window, mouse_callback);
	//glfwSetMouseButtonCallback(window, mouse_button_callback);
	

	// Initialize OpenGL loader
	bool err = glewInit() != GLEW_OK;
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.49f, 0.49f, 0.49f, 1.00f); //gray

	glEnable(GL_DEPTH_TEST); 
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	Shader shader("../Shaders/FaceVertex2.vs", "../Shaders/FaceFragment.fs");

	std::set<std::string> filesDir;
	const std::string dir = "../Models/High";
	get_files_in_directory(filesDir, dir);

	for (std::set<std::string>::iterator it = filesDir.begin(); it != filesDir.end(); ++it)
		std::cout << ' ' << *it << '\n';
	std::cout << filesDir.size() << '\n';


	unsigned int teapotVAO[25];
	std::vector<glm::vec3> teapotVertices[25];
	std::vector<glm::vec2> teapotTextures[25];
	std::vector<glm::vec3> teapotNormals[25];
	GLuint vertexbuffer[25];
	GLuint normbuffer[25];

	
	GLuint ivertexbuffer[25];
	GLuint inormbuffer[25];
	GLuint indexbuffer[25];

	unsigned int sphereVAO;
	std::vector<glm::vec3> sphereVertices;
	std::vector<glm::vec2> sphereTextures;
	std::vector<glm::vec3> sphereNormals;


	
	for (int i = 0; i<nrFaces; i++) {
		
		// teapot VAO
		glGenVertexArrays(1, &teapotVAO[i]);
		glBindVertexArray(teapotVAO[i]);

		//load object
		std::string ObjName = *std::next(filesDir.begin(), i);
		std::string FileLoc = "../Models/High/";
		FileLoc.append(ObjName);
		bool res = loadOBJ(FileLoc.c_str(), teapotVertices[i], teapotTextures[i], teapotNormals[i]);
		//bool res = read_obj("../Models/neutral.obj", teapotVertices, teapotTextures, teapotNormals, indices);

		//for indexes..
		
		indexVBO_TBN(teapotVertices[i], teapotTextures[i], teapotNormals[i],
			indices[i], indexed_vertices[i], indexed_uvs[i], indexed_normals[i]);

		//std::cout << "face " << i << "--> " << indexed_vertices[i] << std::endl;
		printf("face %d --> vertices %d \n", i, indexed_vertices[i].size());

		glGenBuffers(1, &ivertexbuffer[i]);
		glBindBuffer(GL_ARRAY_BUFFER, ivertexbuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, indexed_vertices[i].size() * sizeof(glm::vec3), &indexed_vertices[i][0], GL_STATIC_DRAW);

		glGenBuffers(1, &inormbuffer[i]);
		glBindBuffer(GL_ARRAY_BUFFER, inormbuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, indexed_normals[i].size() * sizeof(glm::vec3), &indexed_normals[i][0], GL_STATIC_DRAW);

		glGenBuffers(1, &indexbuffer[i]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices[i].size() * sizeof(unsigned short), &indices[i][0], GL_STATIC_DRAW);
		
	}
	
	for (int i = 0; i < nrFaces; i++) {
		
		glEnableVertexAttribArray(i * 2);
		glBindBuffer(GL_ARRAY_BUFFER, ivertexbuffer[i]);
		glVertexAttribPointer(i * 2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glEnableVertexAttribArray(i * 2 + 1);
		glBindBuffer(GL_ARRAY_BUFFER, inormbuffer[i]);
		glVertexAttribPointer(i * 2 + 1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	}
//sphere obj not working----------------------------------------------------------------------------------------------------------------//
	// teapot VAO
	////GLuint sphereVAO;
	//glGenVertexArrays(1, &sphereVAO);
	//glBindVertexArray(sphereVAO);

	////load object
	//bool res = loadOBJ("../Models/earth.obj", sphereVertices, sphereTextures, sphereNormals);

	//
	//GLuint svertexbuffer;
	//glGenBuffers(1, &svertexbuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, svertexbuffer);
	//glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(glm::vec3), &sphereVertices[0], GL_STATIC_DRAW);

	//GLuint snormbuffer;
	//glGenBuffers(1, &snormbuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, snormbuffer);
	//glBufferData(GL_ARRAY_BUFFER, sphereNormals.size() * sizeof(glm::vec3), &sphereNormals[0], GL_STATIC_DRAW);

	//glEnableVertexAttribArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, svertexbuffer);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	//glEnableVertexAttribArray(1);
	//glBindBuffer(GL_ARRAY_BUFFER, snormbuffer);
	//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
//----------------------------------------------------------------------------------------------------------------------------------//

	//adding vetices and normals to Eigen matrix
	nrxyzVertices = indexed_vertices[0].size()*3;
	nrIndices = indexed_vertices[0].size();
	//-----------------------------------------------------------------------------------------------------------------------
	//debug
	printf("nrxyzVertices %d\nindexed_vertices[0].size() %d\n ", nrxyzVertices, indexed_vertices[0].size());


	MatrixXf facesXfv(nrFaces, nrxyzVertices);

	for (int j = 0; j < nrFaces; j++) {
		for (int i = 0; i < nrIndices; i++) {
			facesXfv(j , i) = indexed_vertices[j][i].x;
		}
		for (int i = 0; i < nrIndices; i++) {
			facesXfv(j , i + nrIndices) = indexed_vertices[j][i].y;
		}
		for (int i = 0; i < nrIndices; i++) {
			facesXfv(j , i + (nrIndices * 2)) = indexed_vertices[j][i].z;
		}
	}

	std::cout <<"value "<< facesXfv(1, 2) << "\n";

	MatrixXf facesXfn(nrFaces, nrxyzVertices);

	for (int j = 0; j < nrFaces; j++) {
		for (int i = 0; i < nrIndices; i++) {
			facesXfn(j , i) = indexed_normals[j][i].x;
		}
		for (int i = 0; i < nrIndices; i++) {
			facesXfn(j , i + nrIndices) = indexed_normals[j][i].y;
		}
		for (int i = 0; i < nrIndices; i++) {
			facesXfn(j , i + (nrIndices * 2)) = indexed_normals[j][i].z;
		}
	}

	//missing file-------------------------------------------------------------------------------------------------------
	double animationStartTime = 0;
	int animationFrameIndex = 0;

	readAnimation("animation.txt", tmpAnimation);
	nrFrames = (tmpAnimation.size() / (nrFaces - 1));
	//runAnimation = true;

	animMatrix = setAnimationWeights(tmpAnimation);
	//missing file end---------------------------------------------------------------------------------------------------


	// Main loop
//--------------------------------------------------------------------------------------------------------------------WHILE
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		

		ImGui::Begin("MERY'S FACE");                          // Create a window called "Hello, world!" and append into it.

		//ImGui::Text("Neutral Face");
		ImGui::SliderFloat("Neutral Face", &f[0], 0.0f, 1.0f);
		ImGui::SliderFloat("Jaw Open", &f[1], 0.0f, 1.0f);
		ImGui::SliderFloat("Kiss", &f[2], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();

		ImGui::SliderFloat("Left Brow Lower", &f[3], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Brow Narrow", &f[4], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Brow Raise", &f[5], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();

		ImGui::SliderFloat("Left Eye Closed", &f[6], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Eye Lower Open", &f[7], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Eye Upper Open", &f[8], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();

		ImGui::SliderFloat("Left Nose Wrinkle", &f[9], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();

		ImGui::SliderFloat("Left Puff", &f[10], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Sad", &f[11], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Smile", &f[12], 0.0f, 1.0f);
		ImGui::SliderFloat("Left Suck", &f[13], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();

		ImGui::SliderFloat("Right Brow Lower", &f[14], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Brow Narrow", &f[15], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Brow Raise", &f[16], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();
				
		ImGui::SliderFloat("Right Nose Wrinkle", &f[17], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();
		ImGui::SliderFloat("Right Eye Closed", &f[18], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Eye Lower Open", &f[19], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Eye Upper Open", &f[20], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();
				
		ImGui::SliderFloat("Right Puff", &f[21], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Sad", &f[22], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Smile", &f[23], 0.0f, 1.0f);
		ImGui::SliderFloat("Right Suck", &f[24], 0.0f, 1.0f);
		ImGui::Spacing(); ImGui::Spacing();
		static int counter = 0;
		if (ImGui::Button("Run Animation")) {                            // Buttons return true when clicked (most widgets return true when edited/activated)
			counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);
			runAnimation = true;
			animationStartTime = glfwGetTime();
			animationFrameIndex = 0;
		}
		if (ImGui::Button("Reset")) {                           // Buttons return true when clicked (most widgets return true when edited/activated)
			counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);
			resetAll();
		}

		ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
			
			
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();
		

		bool change = false;
		double animationframegap = 0.2;
		//missing file--------------------------------------------------------------------------------------------------------------
		if (runAnimation) {
			int colCount = nrFaces - 1;
			int rowCount = nrFrames;
			
				for (int j = 0; j < colCount; j++)
				{
					f[j+1] = animMatrix(animationFrameIndex, j);
				}

				//delay
				double currentTime = glfwGetTime();
				double delta = currentTime - animationStartTime;
				std::cout << "delta : " << delta << std::endl;
				if (delta > animationframegap)
				{
					animationStartTime = currentTime;
					animationFrameIndex++;
				}
				
				if (animationFrameIndex >= rowCount)
					runAnimation = false;
		}
		//---------------------------------------------------------------------------------------------

		int numRows = animMatrix.rows();
		int numCols = animMatrix.cols();

		VectorXf weightsXf(nrFaces);
		f[0] = 1;
		for (int i = 0; i < nrFaces; i++) {
			weightsXf(i) = f[i];
		}
		

		
		int display_w, display_h;
		glfwMakeContextCurrent(window);
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(500, 0, display_w, display_h);

		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		

		//calculate final face
		VectorXf deltaBlendXfv(nrxyzVertices);
		deltaBlendXfv = blendFaces(/*neutralXfv,*/ facesXfv, weightsXf/*, neutralf*/);
		

		//new buffer vertex pos
		for (int i = 0; i < nrIndices; i++)
			indexed_vertices[0][i].x = deltaBlendXfv(i);
		for (int i = 0; i < nrIndices; i++)
			indexed_vertices[0][i].y = deltaBlendXfv(i + nrIndices);
		for (int i = 0; i < nrIndices; i++)
			indexed_vertices[0][i].z = deltaBlendXfv(i + (nrIndices * 2));

		
		for (int i = 0; i < nrFaces; i++)
			glBindVertexArray(teapotVAO[i]);

		glGenBuffers(1, &ivertexbuffer[0]);
		glBindBuffer(GL_ARRAY_BUFFER, ivertexbuffer[0]);
		glBufferData(GL_ARRAY_BUFFER, indexed_vertices[0].size() * sizeof(glm::vec3), &indexed_vertices[0][0], GL_STATIC_DRAW);

		

	//normals
		VectorXf deltaBlendXfn(nrxyzVertices);
		deltaBlendXfn = blendFaces(/*neutralXfn, */facesXfn, weightsXf/*, neutralf*/);

		//new buffer normals pos
		for (int i = 0; i < nrIndices; i++)
			indexed_normals[0][i].x = deltaBlendXfn(i);
		for (int i = 0; i < nrIndices; i++)
			indexed_normals[0][i].y = deltaBlendXfn(i + nrIndices);
		for (int i = 0; i < nrIndices; i++)
			indexed_normals[0][i].z = deltaBlendXfn(i + (nrIndices * 2));

		glGenBuffers(1, &inormbuffer[0]);
		glBindBuffer(GL_ARRAY_BUFFER, inormbuffer[0]);
		glBufferData(GL_ARRAY_BUFFER, indexed_normals[0].size() * sizeof(glm::vec3), &indexed_normals[0][0], GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, ivertexbuffer[0]);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, inormbuffer[0]);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

			//draw teapot
		modelTeapot = glm::mat4(1.0f);
		viewTeapot = glm::lookAt(glm::vec3(0.0f, 0.0f, 7.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		projTeapot = glm::perspective(glm::radians(45.0f), (float)wWidth / (float)wHeight, 0.1f, 100.0f);
		modelTeapot = glm::scale(modelTeapot, glm::vec3(0.1f, 0.1f, 0.1f));
		modelTeapot = glm::translate(modelTeapot, glm::vec3(0.0f, -15.0f, 0.0f));

		glm::vec3 lightPos = glm::vec3(0, 0, 8);

		shader.use();
		shader.setMat4("proj", projTeapot);
		shader.setMat4("view", viewTeapot);
		shader.setMat4("model", modelTeapot);
		shader.setVec3("LightPosW", lightPos);
		

		for(int i=0; i<nrFaces; i++)
			glBindVertexArray(teapotVAO[i]);

		for(int i=0; i<nrFaces; i++)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer[i]);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, indices[0].size(), GL_UNSIGNED_SHORT, (void*)0);

		//draw sphere
		//sphere not working---------------------------------------------------------------------------------------//
		/*glm::mat4 modelSphere = glm::mat4(1.0f);
		modelSphere = glm::scale(modelSphere, glm::vec3(0.0005f, 0.0005f, 0.0005f));
		if (inters) {
			modelSphere = glm::translate(modelSphere, glm::vec3(indexed_vertices[0][tindex]));
		}
		
		shader.use();
		shader.setMat4("model", modelSphere);

		glBindVertexArray(sphereVAO);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, svertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, snormbuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glDrawArrays(GL_TRIANGLES, 0, sphereVertices.size());
		glBindVertexArray(0);*/
		//------------------------------------------------------------------------------------------------------------------------//


		// Rendering
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwMakeContextCurrent(window);
		glfwSwapBuffers(window);
	}
	//---------------------------------------------------------------------------------------------------------END WHILE


	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	xmouse = xpos;
	ymouse = ypos;
	//std::cout << "mouse " << xmouse << " " << ymouse << "\n";
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	//if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {

	//			ray = mousePicking(projTeapot, viewTeapot);
	//			//pick indexes 3x3
	//			bool foundIntersect = false;
	//			glm::vec3 intersectPoint;
	//			allIntersects.clear();
	//			std::vector<float> tVector;
	//			std::vector<glm::vec3> intersectedTriangles;
	//			std::vector<int> indexesIntersectedTri;

	//			for (int i = 0; i < indices[0].size(); i += 3) {
	//				glm::vec3 Triangle;
	//				Triangle.x = i;
	//				Triangle.y = i + 1;
	//				Triangle.z = i + 2;

	//				glm::vec4 P14 =
	//					modelTeapot *
	//					glm::vec4(
	//						indexed_vertices[0][Triangle.x].x,
	//						indexed_vertices[0][Triangle.x].y,
	//						indexed_vertices[0][Triangle.x].z,
	//						1.0);
	//				glm::vec4 P24 =
	//					modelTeapot *
	//					glm::vec4(
	//						indexed_vertices[0][Triangle.y].x,
	//						indexed_vertices[0][Triangle.y].y,
	//						indexed_vertices[0][Triangle.y].z,
	//						1.0);
	//				glm::vec4 P34 =
	//					modelTeapot *
	//					glm::vec4(
	//						indexed_vertices[0][Triangle.z].x,
	//						indexed_vertices[0][Triangle.z].y,
	//						indexed_vertices[0][Triangle.z].z,
	//						1.0);

	//				glm::vec3 P1(P14);
	//				glm::vec3 P2(P24);
	//				glm::vec3 P3(P34);

	//				
	//				foundIntersect = RayIntersectsTriangle(
	//					glm::vec3(0.0f, 0.0f, 7.0f),
	//					ray,
	//					P1,
	//					P2,
	//					P3,
	//					intersectedTriangles,
	//					//intersectedTriangles,
	//					intersectPoint,
	//					tVector);

	//				if (foundIntersect) {
	//					indexesIntersectedTri.push_back(i);
	//				}
	//			}
	//				

	//			int tempMinDist = tVector[0];
	//			int posOfMin = 0;
	//			if (allIntersects.size() != 0) {
	//					for (int i = 1; i < tVector.size(); i++) {
	//						if (tempMinDist < tVector[i]) {
	//							tempMinDist = tVector[i];
	//							posOfMin = i;
	//						}

	//					}

	//					tindex = indexesIntersectedTri[posOfMin];
	//					inters = true;
	//				

	//				
	//			}

	//			std::cout << "intersect" << allIntersects.size() << "\n";
	//			for (int i = 0; i < allIntersects.size(); i++) {
	//				std::cout<<i<<"-------"<<allIntersects[i].x<<" "<< allIntersects[i].y<<" "<< allIntersects[i].z <<"\n";
	//			}

	//			std::cout << "final intersect index " << tindex << "\n";
	//			std::wcout << "final triangle "
	//				<< indexed_vertices[0][tindex].x << " "
	//				<< indexed_vertices[0][tindex].y << " "
	//				<< indexed_vertices[0][tindex].z << "\n";
	//			std::cout << "something\n";


	//}
}