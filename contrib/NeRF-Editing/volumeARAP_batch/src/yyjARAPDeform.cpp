#include <Eigen/SVD>
#include <omp.h>
#include <ctime>
#include "ARAPDeform.h"

using namespace std;
#define DUSE_OPENMP

// #ifdef DUSE_OPENMP
// #define OMP_open __pragma(omp parallel num_threads(omp_get_num_procs()*2)) \
// { \
// 	__pragma(omp for)
// #define OMP_end \
// }
// #else
// #define OMP_open ;
// #define OMP_end ;
// #endif

double yyj_cotan(Eigen::Vector3d a, Eigen::Vector3d b) {
	double na = a.norm(), nb = b.norm();
	if (na < Eps || nb < Eps) return 0;
	double cos = a.dot(b) / (na*nb);
	if (cos == 1) return 1;
	return cos / sqrt(1 - cos * cos);
}

Eigen::Matrix3d vec2mat(Eigen::Vector3d a, Eigen::Vector3d b) {
	Eigen::Matrix3d resultMat;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			resultMat(i, j) = a[i] * b[j];
		}
	}
	return resultMat;
}

double calDihedralAngle(std::vector<OpenVolumeMesh::Geometry::Vec3d> dihedral) {
    // v1, v2 constructs the common edge. v3 and v4 expand the dihedral angle
    OpenVolumeMesh::Geometry::Vec3d v1 = dihedral[0];
    OpenVolumeMesh::Geometry::Vec3d v2 = dihedral[1];
    OpenVolumeMesh::Geometry::Vec3d v3 = dihedral[2];
    OpenVolumeMesh::Geometry::Vec3d v4 = dihedral[3];

    // assume the order: v3 -> v1 -> v2 -> v4
    OpenVolumeMesh::Geometry::Vec3d d1 = v1 - v3;
    OpenVolumeMesh::Geometry::Vec3d d2 = v2 - v1;
    OpenVolumeMesh::Geometry::Vec3d d3 = v4 - v2;

    OpenVolumeMesh::Geometry::Vec3d n1 = d1.cross(d2);
    OpenVolumeMesh::Geometry::Vec3d n2 = d2.cross(d3);

    double cos_angle = n1.dot(n2) / n1.norm() / n2.norm();

    return cos_angle / sqrt(1 - cos_angle*cos_angle);
}

ARAPDeform::ARAPDeform(TetrahedralMesh& input_mesh, bool hardConstrain) :mesh(&input_mesh)
{
	degree.resize(input_mesh.n_vertices(), 0);
	edge_index.resize(input_mesh.n_vertices(), 0);
	half_edge_num = 0;
	for (int i = 0; i < input_mesh.n_vertices(); i++)
	{
		VertexHandle vi(i);
		std::vector<Tet_vec3d> neighborPoints;
		Tet_vec3d pi = input_mesh.vertex(vi);

		edge_index[i] = half_edge_num;
		for (OpenVolumeMesh::VertexVertexIter vv_it = input_mesh.vv_iter(vi); vv_it.valid(); ++vv_it)
		{
			int j = vv_it->idx();
			neighborPoints.push_back(input_mesh.vertex(VertexHandle(j)));
			half_edge_num++;
			degree[i]++;
		}
		

		int neighborsNum = (int)neighborPoints.size();

		std::vector<double> weights(neighborsNum, 0);

		// calculate e dihedral angles
		int cnt = 0;
        for (OpenVolumeMesh::VertexOHalfEdgeIter voheit=input_mesh.voh_iter(vi); voheit.valid(); voheit++) {
            OpenVolumeMesh::VertexHandle vi2 = input_mesh.to_vertex_handle(*voheit);
            // std::cout << "****** the edge between ****** " << vi.idx() << " and " << vi2.idx() << std::endl;
			int cell_num = 0;
            OpenVolumeMesh::EdgeHandle eh = input_mesh.edge_handle(*voheit);
            for (OpenVolumeMesh::EdgeCellIter ecit=input_mesh.ec_iter(eh); ecit.valid(); ecit++) {
                // std::cout << "iterate cells ..." << std::endl;
                std::vector<OpenVolumeMesh::Geometry::Vec3d> dihedralAngle;
                // vector size is 4. The first and second is the common edge
                for (OpenVolumeMesh::CellVertexIter cvit=input_mesh.cv_iter(*ecit); cvit.valid(); cvit++) {
                    if (*cvit != vi && *cvit != vi2) {
                        dihedralAngle.push_back(input_mesh.vertex(*cvit));
                        // std::cout << "verts add " << (*cvit).idx() << std::endl;
                    }
                }
                dihedralAngle.push_back(input_mesh.vertex(vi));
                dihedralAngle.push_back(input_mesh.vertex(vi2));
                // std::cout << "specific verts are: " << vi.idx() << ", " << vi2.idx() << std::endl;
				cell_num++;
				weights[cnt] += calDihedralAngle(dihedralAngle);
                // std::cout << "the cot of dihedralAngle is: " << calDihedralAngle(dihedralAngle) << std::endl;
            }
			weights[cnt] /= cell_num; cnt++;
        }
		assert(cnt == neighborsNum);

		for (int neighborCounts = 0; neighborCounts < neighborsNum; neighborCounts++)
		{
			Tet_vec3d edgeij_tet = pi - neighborPoints[neighborCounts];
			Eigen::Vector3d edgeij = OVtoE(edgeij_tet);
			this->edgeijs.push_back(edgeij);

			/*double w1 = yyj_cotan(OVtoE(pi - neighborPoints[(neighborCounts + neighborsNum - 1) % neighborsNum]), OVtoE(neighborPoints[neighborCounts] - neighborPoints[(neighborCounts + neighborsNum - 1) % neighborsNum]));
			double w2 = yyj_cotan(OVtoE(pi - neighborPoints[(neighborCounts + 1) % neighborsNum]), OVtoE(neighborPoints[neighborCounts] - neighborPoints[(neighborCounts + 1) % neighborsNum]));
			weights[neighborCounts] = 0.5*(w1 + w2);*/
			// weights[neighborCounts] = 1;
			/*if ((weights[neighborCounts] != weights[neighborCounts]) || weights[neighborCounts] > 100000 || abs(weights[neighborCounts]) < Eps)
			{

			}*/
			edge_weights.push_back(weights[neighborCounts]);
		}
	}

	isConst.resize(input_mesh.n_vertices(), false);
	isConst_i.resize(input_mesh.n_vertices(), 0);
	//constPoint.resize(input_mesh.n_vertices(), Eigen::Vector3d(0, 0, 0));
	control_index.resize(input_mesh.n_vertices());
	control_weight.resize(input_mesh.n_vertices());
	//this->matEngine = Utility::MatEngine();
	//this->matEngine.OpenEngine();

	this->AcsrRowIndPtr = NULL;
	this->AcsrColPtr = NULL;
	this->AcsrValPtr = NULL;
	this->AcsrColNum = 0;
	this->AcsrRowNum = 0;
	this->Annz = 0;
	this->vectorBPtr = NULL;
	this->vectorBSize = 0;
	this->resultX = NULL;

	maxIterTime = 10;
	this->hardConstrain = hardConstrain;

}

void ARAPDeform::setConstPoint(int i, Eigen::Vector3d v) {
	//isConst[i] = true;
	//isConst_i[i] = 1;
	// constPoint.push_back(v);
	seq_constPoint[i].push_back(v);
}

void ARAPDeform::loadConstPoint(std::istream& cin) {
	int n;
	cin >> n; // sequence³¤¶È
	seq_constPoint.resize(n);
	for (int i = 0; i < n; i++) {
		int m; // Ã¿¸ö¿ØÖÆµã¸öÊý
		cin >> m;
		std::cout << "loading " << m << " control points\n";
		std::vector<int> ids(m);
		for (int j = 0; j < m; j++) {
			double x, y, z;
			cin >> x >> y >> z;
			this->setConstPoint(i, Eigen::Vector3d(x, y, z));
		}
	}
	// add barycentric as constrain
	cin >> n; double u, v, w, z; int v1, v2, v3, v4;
	std::cout << "loading " << n << " barycentric coordinates and tetrahedral vertex index\n";
	for (int i = 0; i < n; i++) {
		cin >> v1 >> v2 >> v3 >> v4;
		bary_vert_index.push_back(Eigen::Vector4i(v1, v2, v3, v4));
		cin >> u >> v >> w >> z;
		barycentric.push_back(Eigen::Vector4d(u, v, w, z));
		control_index[v1].push_back(i);
		control_index[v2].push_back(i);
		control_index[v3].push_back(i);
		control_index[v4].push_back(i);
		control_weight[v1].push_back(u);
		control_weight[v2].push_back(v);
		control_weight[v3].push_back(w);
		control_weight[v4].push_back(z);
	}
}

void ARAPDeform::global_step_pre(TetrahedralMesh& deformedMesh)
{
	/*int vvpairs = 0;
	for (int i = 0; i < mesh->n_vertices(); i++)
	{
	VertexHandle vi(i);
	for (OpenVolumeMesh::VertexVertexIter vvi = mesh->vv_iter(vi); vvi.valid(); vvi++)
	{
	Eigen::Vector3d q = OVtoE(mesh->vertex(VertexHandle(vvi->idx())) - mesh->vertex(vi));
	}
	}*/
	int m = 0;
	for (int i = 0; i < mesh->n_vertices(); i++)
	{
		m += 3 * this->degree[i];
	}
	//cout << mesh->n_edges() << " " << 3 * half_edge_num;
	//m += this->controlpoint_number.size() * 3;
	int row_num = m + this->controlpoint_number.size() * 3;
	if (this->AcsrRowIndPtr == NULL)
	{
		this->AcsrRowIndPtr = (int *)malloc(sizeof(int)*(row_num + 1));
		this->AcsrRowIndPtr[0] = 0;
	}

	int ele_num;
	if (!hardConstrain)
	{
		ele_num = m * 2 + this->controlpoint_number.size() * 3 * 4;
	}
	else
	{
		int n = 0;
		for (int i = 0; i < control_index.size(); i++)
		{
			n += 3 * this->degree[i] * control_index[i].size();
		}
		ele_num = m * 2 + this->controlpoint_number.size() * 3 * 4 + n;
	}
	if (this->AcsrColPtr == NULL)
	{
		this->AcsrColPtr = (int *)malloc(sizeof(int) * ele_num);
	}
	if (this->AcsrValPtr == NULL)
	{
		this->AcsrValPtr = (double *)malloc(sizeof(double) * ele_num);
		memset(this->AcsrValPtr, 0, sizeof(double) * ele_num);
	}
	this->vectorBSize = row_num + 1;
	this->vectorBPtr = (double *)malloc(vectorBSize * sizeof(double));

	if (!hardConstrain)
	{
		this->resultX = (double *)malloc(mesh->n_vertices() * 3 * sizeof(double));
		memset(this->resultX, 0, sizeof(double)*mesh->n_vertices() * 3);
	}
	else
	{
		this->resultX = (double *)malloc((mesh->n_vertices() * 3 + this->controlpoint_number.size()) * sizeof(double));
		memset(this->resultX, 0, sizeof(double)*(mesh->n_vertices() * 3 + this->controlpoint_number.size()));
	}

	int csrRowIndPtrCounter = 1;
	int csrColPtrCounter = 0;
	int csrValAssignCounter = 0;
	int maxCol = 0;

	// Deformation Term //set rowIndPtr and colPtr for Matrix A in Ax=b
	for (int i = 0; i < mesh->n_vertices(); i++)   //base energy item
	{
		VertexHandle vi(i);
		for (OpenVolumeMesh::VertexVertexIter vj = deformedMesh.vv_iter(vi); vj; vj++)
		{
			int j = vj->idx();
			for (int index = 0; index < 3; index++)//x,y,z,3 axis
			{
				int handlePointCounter = 0;
				if (!this->isConst[i] && this->isConst[j])//handle Point
				{
					this->AcsrColPtr[csrColPtrCounter++] = i * 3 + index;
					handlePointCounter = 1;
				}
				else if (this->isConst[i] && !this->isConst[j])
				{
					this->AcsrColPtr[csrColPtrCounter++] = j * 3 + index;
					handlePointCounter = 1;
				}
				else if (!this->isConst[i] && !this->isConst[j])
				{
					if (i > j)//point coefficient
					{
						this->AcsrColPtr[csrColPtrCounter++] = j * 3 + index;
						this->AcsrColPtr[csrColPtrCounter++] = i * 3 + index;
					}
					else {
						this->AcsrColPtr[csrColPtrCounter++] = i * 3 + index;
						this->AcsrColPtr[csrColPtrCounter++] = j * 3 + index;
					}
				}
				else if (this->isConst[i] && this->isConst[j])
				{
					handlePointCounter = 2;
				}
				if (hardConstrain)
				{
					for (int k = 0; k < control_index[i].size(); k++)
					{
						this->AcsrColPtr[csrColPtrCounter++] = mesh->n_vertices() * 3 + control_index[i][k];
					}
					//set rowIndex
					this->AcsrRowIndPtr[csrRowIndPtrCounter++] = 2 - handlePointCounter + this->AcsrRowIndPtr[csrRowIndPtrCounter - 1] + control_index[i].size();
				}
				else
				{
					this->AcsrRowIndPtr[csrRowIndPtrCounter++] = 2 - handlePointCounter + this->AcsrRowIndPtr[csrRowIndPtrCounter - 1];
				}
			}
		}
	}
	std::cout << "the lapalce part has " << csrRowIndPtrCounter - 1 << " rows" << std::endl;
	std::cout << "the lapalce part has " << csrColPtrCounter << " non-zero elements" << std::endl;
	//Handle point,set rowIndPtr and colPtr
	for (int i = 0; i < this->controlpoint_number.size(); i++)
	{
		//std::cout << "i: " << i << " csrColPtrCounter: " << csrColPtrCounter << ", value:" << bary_vert_index[this->controlpoint_number[i].first] << std::endl;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][0] * 3 + 0;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][1] * 3 + 0;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][2] * 3 + 0;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][3] * 3 + 0;

		this->AcsrRowIndPtr[csrRowIndPtrCounter++] = 4 + this->AcsrRowIndPtr[csrRowIndPtrCounter - 1];

		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][0] * 3 + 1;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][1] * 3 + 1;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][2] * 3 + 1;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][3] * 3 + 1;

		this->AcsrRowIndPtr[csrRowIndPtrCounter++] = 4 + this->AcsrRowIndPtr[csrRowIndPtrCounter - 1];

		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][0] * 3 + 2;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][1] * 3 + 2;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][2] * 3 + 2;
		this->AcsrColPtr[csrColPtrCounter++] = bary_vert_index[this->controlpoint_number[i].first][3] * 3 + 2;

		this->AcsrRowIndPtr[csrRowIndPtrCounter++] = 4 + this->AcsrRowIndPtr[csrRowIndPtrCounter - 1];

	}

	for (int i = 0; i < mesh->n_vertices(); i++)
	{
		VertexHandle vi(i);
		//modify ±È½ÏÀ÷º¦µÄr
		//Eigen::Matrix3d &ri = this->featurevector_result.rots[i].r;
		//Eigen::Matrix3d &ri = R[i];
		int edgeijIndex = this->edge_index[i];//first neighbour edge index
											  //iterate point j (i adjacent points)
		for (OpenVolumeMesh::VertexVertexIter vj = deformedMesh.vv_iter(vi); vj; vj++)
		{
			int j = vj->idx();
			double lambdaDeformWeightCiCij = 1.0 * (this->edge_weights[edgeijIndex]);
			for (int axis = 0; axis < 3; axis++)//x,y,z
			{
				//Point parameter
				if (!this->isConst[i] && !this->isConst[j])
				{
					if (i < j)//point coefficient
					{
						this->AcsrValPtr[csrValAssignCounter++] = lambdaDeformWeightCiCij;
						this->AcsrValPtr[csrValAssignCounter++] = -lambdaDeformWeightCiCij;
					}
					else
					{
						this->AcsrValPtr[csrValAssignCounter++] = -lambdaDeformWeightCiCij;
						this->AcsrValPtr[csrValAssignCounter++] = lambdaDeformWeightCiCij;
					}
				}
				else if (!this->isConst[i] && this->isConst[j])
				{
					this->AcsrValPtr[csrValAssignCounter++] = lambdaDeformWeightCiCij;
				}
				else if (this->isConst[i] && !this->isConst[j])
				{
					this->AcsrValPtr[csrValAssignCounter++] = -lambdaDeformWeightCiCij;
				}
				if (hardConstrain)
				{
					for (int k = 0; k < control_weight[i].size(); k++)
					{
						this->AcsrValPtr[csrValAssignCounter++] = control_weight[i][k] / 4;
					}
				}
			}//end of xyz
			edgeijIndex++;
		}//end of j
	}//end of i

	 //handle point as hard constrain
	for (int i = 0; i < this->controlpoint_number.size(); i++)
	{
		for (int j = 0; j < 3; j++) {
			this->AcsrValPtr[csrValAssignCounter++] = barycentric[i][0];
			this->AcsrValPtr[csrValAssignCounter++] = barycentric[i][1];
			this->AcsrValPtr[csrValAssignCounter++] = barycentric[i][2];
			this->AcsrValPtr[csrValAssignCounter++] = barycentric[i][3];
		}
	}
	std::cout << "the all maxtrix has " << csrRowIndPtrCounter - 1 << " rows" << std::endl;
	std::cout << "the all maxtrix has " << csrValAssignCounter << " non-zero elems" << std::endl;
}

void ARAPDeform::eigen_global_step_pre(TetrahedralMesh& deformedMesh)
{
	int row_num = (half_edge_num + this->controlpoint_number.size()) * 3;

	int ele_num;
	if (!hardConstrain)
	{
		ele_num = half_edge_num * 3 * 2 + this->controlpoint_number.size() * 3 * 4;
	}
	else
	{
		int n = 0;
		for (int i = 0; i < control_index.size(); i++)
		{
			n += 3 * this->degree[i] * control_index[i].size();
		}
		ele_num = half_edge_num * 3 * 2 + this->controlpoint_number.size() * 3 * 4 + n;
	}
	this->tripletList.reserve(ele_num);
	this->vectorBSize = row_num;
	this->vectorBPtr = (double *)malloc(vectorBSize * sizeof(double));

	int rowCounter = 0;
	int axisNum = 3;

	// Deformation Term //set rowIndPtr and colPtr for Matrix A in Ax=b
	for (int i = 0; i < mesh->n_vertices(); i++)   //base energy item
	{
		VertexHandle vi(i);
		int edgeijIndex = this->edge_index[i];//first neighbour edge index
		for (OpenVolumeMesh::VertexVertexIter vj = deformedMesh.vv_iter(vi); vj; vj++)
		{
			int j = vj->idx();
			double lambdaDeformWeightCiCij = 1.0 * (this->edge_weights[edgeijIndex]);
			for (int index = 0; index < axisNum; index++)//x,y,z,3 axis
			{
				int handlePointCounter = 0;
				if (!this->isConst[i] && this->isConst[j])//handle Point
				{
					tripletList.push_back(Tri(rowCounter + index, i * 3 + index, lambdaDeformWeightCiCij));
				}
				else if (this->isConst[i] && !this->isConst[j])
				{
					tripletList.push_back(Tri(rowCounter + index, j * 3 + index, -lambdaDeformWeightCiCij));
				}
				else if (!this->isConst[i] && !this->isConst[j])
				{
					if (i > j)//point coefficient
					{
						tripletList.push_back(Tri(rowCounter + index, j * 3 + index, -lambdaDeformWeightCiCij));
						tripletList.push_back(Tri(rowCounter + index, i * 3 + index, lambdaDeformWeightCiCij));
					}
					else {
						tripletList.push_back(Tri(rowCounter + index, i * 3 + index, lambdaDeformWeightCiCij));
						tripletList.push_back(Tri(rowCounter + index, j * 3 + index, -lambdaDeformWeightCiCij));
					}
				}
				if (hardConstrain)
				{
					for (int k = 0; k < control_index[i].size(); k++)
					{
						tripletList.push_back(Tri(rowCounter + index, mesh->n_vertices() * 3 + control_index[i][k], control_weight[i][k] / 4));
					}
				}
			}//end of xyz
			edgeijIndex++;
			rowCounter += axisNum;
		}//end of vj
	}//end of vi
	
	 //Handle point,set rowIndPtr and colPtr
	for (int i = 0; i < this->controlpoint_number.size(); i++)
	{
		for (int index = 0; index < axisNum; index++)
		{
			tripletList.push_back(Tri(rowCounter, bary_vert_index[this->controlpoint_number[i].first][0] * 3 + index, barycentric[i][0]));
			tripletList.push_back(Tri(rowCounter, bary_vert_index[this->controlpoint_number[i].first][1] * 3 + index, barycentric[i][1]));
			tripletList.push_back(Tri(rowCounter, bary_vert_index[this->controlpoint_number[i].first][2] * 3 + index, barycentric[i][2]));
			tripletList.push_back(Tri(rowCounter, bary_vert_index[this->controlpoint_number[i].first][3] * 3 + index, barycentric[i][3]));
			rowCounter++;
		}

	}
}

void ARAPDeform::local_step(std::vector<Eigen::Matrix3d>& R, TetrahedralMesh& deformedMesh)
{
	std::cout << "Local Step" << endl;
	//OMP_open
	for (int i = 0, edgeCounter = 0; i < mesh->n_vertices(); i++)
	{
		Eigen::Matrix3d edgeMatrixSum = Eigen::Matrix3d::Zero();
		VertexHandle vi(i);
		for (OpenVolumeMesh::VertexVertexIter vj = deformedMesh.vv_iter(vi); vj; vj++)
		{
			int j = vj->idx();
			Eigen::Vector3d deformedEdgeij = OVtoE(deformedMesh.vertex(vi) - deformedMesh.vertex(VertexHandle(j)));
			//edgeMatrixSum += edge_weights[edgeCounter] * vec2mat(edgeijs[edgeCounter], deformedEdgeij);
			edgeMatrixSum += vec2mat(edgeijs[edgeCounter], deformedEdgeij);
			edgeCounter++;
		}
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(edgeMatrixSum, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd U = svd.matrixU();
		Eigen::MatrixXd V = svd.matrixV();
		Eigen::Matrix3d tmpR = V * U.transpose();
		if (tmpR.determinant() < 0)
		{
			Eigen::VectorXd sigularDiagonal = svd.singularValues().diagonal();
			double minSigularValue = sigularDiagonal.minCoeff();
			int id = 0;
			for (int m = 0; m < sigularDiagonal.size(); m++)
			{
				if (minSigularValue == sigularDiagonal[m])
					id = m;
			}
			for (int m = 0; m < 3; m++)
			{
				U(m, id) = 0 - U(m, id);
			}
			tmpR = V * U.transpose();
		}
		R[i] = tmpR;
	}
	//OMP_end
}

void ARAPDeform::yyj_ARAPDeform(std::string &handlefile, std::string outputFolder)
{
	std::ifstream iff(handlefile.c_str());
	this->loadConstPoint(iff);
	//for (int i = 0; i < mesh->n_vertices(); i++) {
	for (int i = 0; i < barycentric.size(); i++) {
		this->controlpoint_number.push_back(make_pair(i, Eigen::Vector3d(0, 0, 0))); // ¿ØÖÆµã(ÖØÐÄ×ø±ê)ÊýÄ¿
	}
	TetrahedralMesh *deformed_mesh;
	deformed_mesh = new TetrahedralMesh(*this->mesh);
	std::vector<Eigen::Matrix3d> Rots;

	for (int i = 0; i < mesh->n_vertices(); i++)
	{
		Rots.push_back(Eigen::Matrix3d::Identity());
	}

	//this->global_step_pre(*deformed_mesh);
	this->eigen_global_step_pre(*deformed_mesh);

	int columnNumber;
	if (!hardConstrain)
	{
		columnNumber = mesh->n_vertices() * 3;
	}
	else
	{
		columnNumber = mesh->n_vertices() * 3 + this->controlpoint_number.size();
	}
	int rowNumber = (edgeijs.size() + this->controlpoint_number.size()) * 3;
	//this->vectorBSize = rowNumber + 1;

	//this->yyj_CholeskyPre(this->matEngine, rowNumber, columnNumber, AcsrRowIndPtr[rowNumber], AcsrRowIndPtr, AcsrColPtr, AcsrValPtr);
	std::cout << "Construct sparse A" << std::endl;
	//Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor> > sparseA(rowNumber, columnNumber, AcsrRowIndPtr[rowNumber], AcsrRowIndPtr, AcsrColPtr, AcsrValPtr);
	Eigen::SparseMatrix<double> sparseA(rowNumber, columnNumber);
	sparseA.setFromTriplets(this->tripletList.begin(), this->tripletList.end());
	std::cout << "Construct sparse AT" << std::endl;
	Eigen::SparseMatrix<double> sparseAT = sparseA.transpose();
	std::cout << "cholesky begin" << std::endl;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(sparseAT*sparseA);

	// modify to sequence deformation.
	// ÔÚload_data´¦¶¨Òåseq_constPoint
	for (int seq_id = 0; seq_id < seq_constPoint.size(); seq_id++) {
		constPoint = seq_constPoint[seq_id];
		std::cout << "processing the " << seq_id << " deformation" << std::endl;
		for (int iterationCounter = 0; iterationCounter < this->maxIterTime; iterationCounter++)
		{
			memset(this->vectorBPtr, 0, this->vectorBSize * sizeof(double));
			int rowCounter = 0;  //used in assign value to Vertex Bs

			double lambdaDeformWeightCiCij = 0;
			Eigen::Vector3d edgeij_weight;
			Eigen::Matrix3d Ri, Rj;
			Eigen::Vector3d RiRjEdgeij;
			std::vector<Eigen::Vector3d> VecRiRjEdgeij;
			/*VecRiRjEdgeij.resize(edgeijs.size(), Eigen::Vector3d::Zero());*/

			for (int i = 0, edgeCounter = 0; i < mesh->n_vertices(); i++)
			{
				//iterate point j (i adjacent points)
				VertexHandle vi(i);
				//int edgeijIndex = this->edge_index[i];//first neighbor edge index
				Ri = Rots[i];

				//iterate point j (i adjacent points)
				for (OpenVolumeMesh::VertexVertexIter vj = deformed_mesh->vv_iter(vi); vj; vj++)
				{
					int j = vj->idx();
					//lambdaDeformWeightCiCij = this->edge_weights[edgeijIndex];
					lambdaDeformWeightCiCij = this->edge_weights[edgeCounter];
					edgeij_weight = this->edge_weights[edgeCounter] * this->edgeijs[edgeCounter];//edgejk = pj -pk
																								 //assert(edgeij_weight == edgeij_weight);
					Rj = Rots[j];
					RiRjEdgeij = 0.5 * (Ri + Rj) * edgeij_weight;
					VecRiRjEdgeij.push_back(RiRjEdgeij);
					//VecRiRjEdgeij[_edgetick] = RiRjEdgeij;
					//_edgetick++;
					//Point parameter
					//this->vectorBPtr[rowCounter + 0] = lambdaDeformWeightCiCij * (this->constPoint[j][0] * this->isConst_i[j] - this->constPoint[i][0] * this->isConst_i[i]);
					//this->vectorBPtr[rowCounter + 1] = lambdaDeformWeightCiCij * (this->constPoint[j][1] * this->isConst_i[j] - this->constPoint[i][1] * this->isConst_i[i]);
					//this->vectorBPtr[rowCounter + 2] = lambdaDeformWeightCiCij * (this->constPoint[j][2] * this->isConst_i[j] - this->constPoint[i][2] * this->isConst_i[i]);
					edgeCounter++;
					rowCounter += 3;
				}
			}

			for (int j = 0; j < half_edge_num; j++)
			{
				this->vectorBPtr[j * 3 + 0] += VecRiRjEdgeij[j][0];
				this->vectorBPtr[j * 3 + 1] += VecRiRjEdgeij[j][1];
				this->vectorBPtr[j * 3 + 2] += VecRiRjEdgeij[j][2];
			}

			//handle point as hard constrain						
			for (int i = 0; i < this->controlpoint_number.size(); i++)
			{
				int constrolpointid = this->controlpoint_number[i].first;
				this->vectorBPtr[rowCounter + 0] = constPoint[constrolpointid][0];
				this->vectorBPtr[rowCounter + 1] = constPoint[constrolpointid][1];
				this->vectorBPtr[rowCounter + 2] = constPoint[constrolpointid][2];
				rowCounter += 3;
			}

			//È±ÉÙÒ»¸öº¯ÊýÀûÓÃÕâÐ©±äÁ¿½øÐÐÇó½â
			long t1 = clock();
			//this->yyj_LeastSquareSolve(this->matEngine, rowNumber, columnNumber, AcsrRowIndPtr[rowNumber], AcsrRowIndPtr, AcsrColPtr, AcsrValPtr, vectorBPtr, resultX);
			//this->yyj_CholeskySolve(this->matEngine, rowNumber, columnNumber, vectorBPtr, resultX);
			vectorB.resize(this->vectorBSize);
			for (int i = 0; i < this->vectorBSize; i++)
			{
				vectorB[i] = vectorBPtr[i];
			}
			Eigen::VectorXd x = chol.solve(sparseAT*vectorB);
			std::cout << "Global Time:" << clock() - t1 << std::endl;

			for (int i = 0; i < mesh->n_vertices(); i++)
			{
				VertexHandle vi(i);
				//Tet_vec3d tmp_vertex(this->resultX[i * 3 + 0], this->resultX[i * 3 + 1], this->resultX[i * 3 + 2]);
				Tet_vec3d tmp_vertex(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
				deformed_mesh->set_vertex(vi, tmp_vertex);
			}

			long t2 = clock();
			local_step(Rots, *deformed_mesh);
			std::cout << "Local Time:" << clock() - t2 << std::endl;
		} // end of iteration

		string file_id = std::to_string(seq_id);
		while (file_id.size() < 4) file_id = "0" + file_id;
		string outputName = outputFolder + "/arap_result_" + file_id +"_.ovm";
		myWriteFile(outputName, *deformed_mesh);
		//this->matEngine.EvalString("close");
	}
	//this->matEngine.EvalString("exit");
}

//bool ARAPDeform::yyj_LeastSquareSolve(Utility::MatEngine &matEngine, int rowNum, int colNum, int Annz, int *rowPtr, int *colPtr, double *valPtr, const double *b, double *x)
//{
//	mxArray *sparseMatrixA = mxCreateSparse(colNum, rowNum, Annz, mxREAL);
//	mwIndex *mxIrA = mxGetIr(sparseMatrixA);
//	mwIndex *mxJcA = mxGetJc(sparseMatrixA);
//	double *mxPrA = mxGetPr(sparseMatrixA);
//	for (int i = 0; i < Annz; i++)
//	{
//		mxIrA[i] = (mwIndex)colPtr[i];
//		mxPrA[i] = (double)valPtr[i];
//	}
//	for (int i = 0; i <= rowNum; i++)
//	{
//		mxJcA[i] = (mwIndex)rowPtr[i];
//	}
//	//long t3 = clock();
//	matEngine.PutVariable("AT", sparseMatrixA);
//	//std::cout << "Put AT:" << clock() - t3 << endl;
//
//	mxArray* bb = mxCreateDoubleMatrix(rowNum, 1, mxREAL);
//	double* bbPtr = mxGetPr(bb);
//	for (int i = 0; i < vectorBSize; i++)
//	{
//		bbPtr[i] = b[i];
//	}
//	//t3 = clock();
//	matEngine.PutVariable("b", bb);
//	//std::cout << "Put b:" << clock() - t3 << endl;
//	//t3 = clock();
//	matEngine.EvalString("A = AT';");
//	//matEngine.EvalString("x = AT*A\\(AT*b);");
//	matEngine.EvalString("x = A\\b;");
//	//std::cout << "Solve:" << clock() - t3 << endl;
//	//t3 = clock();
//	mxArray* x1 = matEngine.GetVariable("x");
//	double* x1Ptr = mxGetPr(x1);
//	for (int i = 0; i < colNum; i++)
//	{
//		x[i] = x1Ptr[i];
//	}
//	//std::cout << "Ger result:" << clock() - t3 << endl;
//
//	return true;
//}

//bool ARAPDeform::yyj_CholeskyPre(Utility::MatEngine &matEngine, int rowNum, int colNum, int Annz, int *rowPtr, int *colPtr, double *valPtr)
//{
//	mxArray *sparseMatrixA = mxCreateSparse(colNum, rowNum, Annz, mxREAL);
//	mwIndex *mxIrA = mxGetIr(sparseMatrixA);
//	mwIndex *mxJcA = mxGetJc(sparseMatrixA);
//	double *mxPrA = mxGetPr(sparseMatrixA);
//	for (int i = 0; i < Annz; i++)
//	{
//		mxIrA[i] = (mwIndex)colPtr[i];
//		mxPrA[i] = (double)valPtr[i];
//	}
//	for (int i = 0; i <= rowNum; i++)
//	{
//		mxJcA[i] = (mwIndex)rowPtr[i];
//	}
//	matEngine.PutVariable("AT", sparseMatrixA);
//
//	long t3 = clock();
//	matEngine.EvalString("A = AT';");
//	matEngine.EvalString("[UA, p] = chol(AT*A);");
//	std::cout << "Cholesky:" << clock() - t3 << endl;
//	mxArray* UAp = matEngine.GetVariable("p");
//	double* UApPtr = mxGetPr(UAp);
//	if (UApPtr[0])
//	{
//		matEngine.EvalString("[LA,UA] = lu(AT*A);");
//		std::cout << "LU" << endl;
//	}
//	else
//	{
//		matEngine.EvalString("LA = UA';");
//	}
//
//	return true;
//}

//bool ARAPDeform::yyj_CholeskySolve(Utility::MatEngine &matEngine, int rowNum, int colNum, const double *b, double *x)
//{
//	mxArray* bb = mxCreateDoubleMatrix(rowNum, 1, mxREAL);
//	double* bbPtr = mxGetPr(bb);
//	for (int i = 0; i < vectorBSize; i++)
//	{
//		bbPtr[i] = b[i];
//	}
//	matEngine.PutVariable("b", bb);
//
//	long t3 = clock();
//	matEngine.EvalString("x_tmp = LA\\(AT*b);");
//	std::cout << "Solve1:" << clock() - t3 << endl;
//	t3 = clock();
//	matEngine.EvalString("x = UA\\x_tmp;");
//	std::cout << "Solve2:" << clock() - t3 << endl;
//	mxArray* x1 = matEngine.GetVariable("x");
//	double* x1Ptr = mxGetPr(x1);
//	for (int i = 0; i < colNum; i++)
//	{
//		x[i] = x1Ptr[i];
//	}
//
//	return true;
//}

ARAPDeform::~ARAPDeform()
{
	//this->matEngine.CloseEngine();
	free(this->AcsrRowIndPtr);
	free(this->AcsrColPtr);
	free(this->AcsrValPtr);
	free(this->vectorBPtr);
}