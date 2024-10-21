#pragma once

//#include "MatEngine.h"
#include "MyUtils.h"
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>

typedef Eigen::Triplet<double> Tri;

class ARAPDeform
{
public:
	TetrahedralMesh * mesh;
	int half_edge_num;
	std::vector<int> degree;
	std::vector<Eigen::Vector3d> edgeijs;  // edgeijs vector
	std::vector<double> edge_weights;
	std::vector<int> edge_index;  // first neighbour edge index
	std::vector<std::pair<int, int>> edge_pairs;
	std::vector<bool> isConst;
	std::vector<int> isConst_i;
	std::vector<Eigen::Vector3d> constPoint;
	std::vector<std::vector<Eigen::Vector3d>> seq_constPoint;
	std::vector<Eigen::Vector4i> bary_vert_index; // save index of tet verts for each control points
	std::vector<Eigen::Vector4d> barycentric; // save barycentric coordinate
	std::vector<std::vector<int>> control_index;
	std::vector<std::vector<double>> control_weight;
	std::vector<std::pair<int, Eigen::Vector3d>> controlpoint_number;
	//Utility::MatEngine matEngine;

	int * AcsrRowIndPtr;
	int * AcsrColPtr;
	double * AcsrValPtr;
	int Annz;     // number of non-zero values in matrix A
	int AcsrRowNum;
	int AcsrColNum;
	double *resultX;
	long vectorBSize;
	double * vectorBPtr;

	// for eigen solve
	std::vector<Tri> tripletList;
	Eigen::VectorXd vectorB;

	int maxIterTime;
	bool hardConstrain;

	ARAPDeform() {};
	ARAPDeform(TetrahedralMesh& mesh, bool hardConstrain = true);
	~ARAPDeform();
	void loadConstPoint(std::istream& cin);
	void setConstPoint(int i, Eigen::Vector3d v);
	void global_step_pre(TetrahedralMesh& deformed_mesh);
	void eigen_global_step_pre(TetrahedralMesh& deformed_mesh);
	void local_step(std::vector<Eigen::Matrix3d>& R, TetrahedralMesh& deformed_mesh);
	void yyj_ARAPDeform(std::string &handlefile, std::string outputFolder);
	//bool yyj_LeastSquareSolve(Utility::MatEngine &matEngine, int rowNum, int colNum, int Annz, int *rowPtr, int *colPtr, double *valPtr, const double *b, double *x);
	//bool yyj_CholeskyPre(Utility::MatEngine &matEngine, int rowNum, int colNum, int Annz, int *rowPtr, int *colPtr, double *valPtr);
	//bool yyj_CholeskySolve(Utility::MatEngine &matEngine, int rowNum, int colNum, const double *b, double *x);
};