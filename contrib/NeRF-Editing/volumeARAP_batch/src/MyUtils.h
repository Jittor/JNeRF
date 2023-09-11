#ifndef MYUTILS_H
#define MYUTILS_H

#include <Eigen/Eigen>
#include <OpenVolumeMesh/Core/OpenVolumeMeshHandle.hh>
#include <OpenVolumeMesh/Core/PropertyDefines.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralGeometryKernel.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMeshTopologyKernel.hh>
#include <OpenVolumeMesh/Mesh/HexahedralMeshTopologyKernel.hh>
#include <OpenVolumeMesh/Geometry/VectorT.hh>
// Include the polyhedral mesh header
#include <OpenVolumeMesh/Mesh/PolyhedralMesh.hh>
#include <OpenVolumeMesh/FileManager/FileManager.hh>

#define Eps 1e-10

typedef OpenVolumeMesh::Geometry::Vec3d Tet_vec3d;
//typedef OpenVolumeMesh::TetrahedralGeometryKernel<Tet_vec3d, OpenVolumeMesh::TetrahedralMeshTopologyKernel> TetrahedralMesh;
//typedef OpenVolumeMesh::GeometryKernel<Tet_vec3d, OpenVolumeMesh::TopologyKernel> PolyhedralMesh;
typedef OpenVolumeMesh::GeometricPolyhedralMeshV3d TetrahedralMesh;

using VertexHandle = OpenVolumeMesh::VertexHandle;
using Parameter = Tet_vec3d;
using Position = Tet_vec3d;
using VertexVertexIter = OpenVolumeMesh::VertexVertexIter;

Tet_vec3d EtoOV(const Eigen::Vector3d &v);
Eigen::Vector3d OVtoE(const Tet_vec3d &v);

Eigen::Matrix3d exp(Eigen::Matrix3d);
Eigen::Matrix3d log(Eigen::Matrix3d);

void trimString(std::string& _string);

bool getCleanLine(std::istream& ifs, std::string& _string, bool _skipEmptyLines = true);

bool myReadFile(const std::string& _filename, TetrahedralMesh& _mesh,
	bool _topologyCheck = true,
	bool _computeBottomUpIncidences = true);

void myWriteFile(const std::string& _filename, TetrahedralMesh& _mesh);

//template <class MeshT>
//void myWriteFile(const std::string& _filename, MeshT& _mesh);

#endif
