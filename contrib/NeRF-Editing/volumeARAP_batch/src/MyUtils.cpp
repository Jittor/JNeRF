#include "MyUtils.h"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace OpenVolumeMesh;

Tet_vec3d EtoOV(const Eigen::Vector3d &v) {
	return Tet_vec3d(v(0), v(1), v(2));
}

Eigen::Vector3d OVtoE(const Tet_vec3d &v) {
	return Eigen::Vector3d(v[0], v[1], v[2]);
}

Eigen::Matrix3d exp(Eigen::Matrix3d x) {
	//return x.exp();
	double theta = sqrt(x(0, 1)*x(0, 1) + x(0, 2)*x(0, 2) + x(1, 2)*x(1, 2));
	if (abs(theta) == 0) return Eigen::Matrix3d::Identity();
	x /= theta;
	return Eigen::Matrix3d::Identity() +
		x * sin(theta) +
		x * x * (1 - cos(theta));
}

Eigen::Matrix3d log(Eigen::Matrix3d x) {
	//return x.log();
	double theta = (x.trace() - 1) / 2;
	theta = std::acos(std::max(-1.0, std::min(1.0, theta)));
	if (abs(theta) == 0) return Eigen::Matrix3d::Zero();
	return (theta / (2 * sin(theta))) * (x - x.transpose());
}

void trimString(std::string& _string) {

	// Trim Both leading and trailing spaces
	size_t start = _string.find_first_not_of(" \t\r\n");
	size_t end = _string.find_last_not_of(" \t\r\n");

	if ((std::string::npos == start) || (std::string::npos == end)) {
		_string = "";
	}
	else {
		_string = _string.substr(start, end - start + 1);
	}
}

bool getCleanLine(std::istream& _ifs, std::string& _string, bool _skipEmptyLines) {

	// While we are not at the end of the file
	while (true) {

		// Get the current line:
		std::getline(_ifs, _string);

		// Remove whitespace at beginning and end
		trimString(_string);

		// Check if string is not empty ( otherwise we continue
		if (_string.size() != 0) {

			// Check if string is a comment ( starting with # )
			if (_string[0] != '#') {
				return true;
			}

		}
		else {
			if (!_skipEmptyLines)
				return true;
		}

		if (_ifs.eof()) {
			std::cerr << "End of file reached while searching for input!" << std::endl;
			return false;
		}
	}

	return false;
}

bool myReadFile(const std::string& _filename, TetrahedralMesh& _mesh,
	bool _topologyCheck, bool _computeBottomUpIncidences)
{

	std::ifstream iff(_filename.c_str(), std::ios::in);

	if (!iff.good()) {
		std::cerr << "Error: Could not open file " << _filename << " for reading!" << std::endl;
		iff.close();
		return false;
	}
	/*return readStream(iff, _mesh, _topologyCheck, _computeBottomUpIncidences);*/

	std::stringstream sstr;
	std::string line;
	std::string s_tmp;
	uint64_t c = 0u;
	typedef typename TetrahedralMesh::PointT Point;
	Point v = Point(0.0, 0.0, 0.0);

	_mesh.clear(false);
	// Temporarily disable bottom-up incidences
	// since it's way faster to first add all the
	// geometry and compute them in one pass afterwards
	_mesh.enable_bottom_up_incidences(false);

	/*
	* Header
	*/

	bool header_found = true;

	// Get first line
	getCleanLine(iff, line);
	sstr.str(line);

	// Check header
	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp != "OVM") {
		//iff.close();
		header_found = false;
		std::cerr << "The specified file might not be in OpenVolumeMesh format!" << std::endl;
		//return false;
	}

	// Get ASCII/BINARY string
	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp == "BINARY") {
		std::cerr << "Binary files are not supported at the moment!" << std::endl;
		return false;
	}

	/*
	* Vertices
	*/
	if (!header_found) {
		sstr.clear();
		sstr.str(line);
	}
	else {
		getCleanLine(iff, line);
		sstr.clear();
		sstr.str(line);
	}

	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp != "VERTICES") {
		std::cerr << "No vertex section defined!" << std::endl;
		return false;
	}
	else {

		// Read in number of vertices
		getCleanLine(iff, line);
		sstr.clear();
		sstr.str(line);
		sstr >> c;

		// Read in vertices
		for (uint64_t i = 0u; i < c; ++i) {

			getCleanLine(iff, line);
			sstr.clear();
			sstr.str(line);
			sstr >> v[0];
			sstr >> v[1];
			sstr >> v[2];
			_mesh.add_vertex(v);
		}
	}

	/*
	* Edges
	*/
	getCleanLine(iff, line);
	sstr.clear();
	sstr.str(line);
	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp != "EDGES") {
		std::cerr << "No edge section defined!" << std::endl;
		return false;
	}
	else {

		// Read in number of edges
		getCleanLine(iff, line);
		sstr.clear();
		sstr.str(line);
		sstr >> c;

		// Read in edges
		for (uint64_t i = 0u; i < c; ++i) {

			unsigned int v1 = 0;
			unsigned int v2 = 0;
			getCleanLine(iff, line);
			sstr.clear();
			sstr.str(line);
			sstr >> v1;
			sstr >> v2;
			_mesh.add_edge(VertexHandle(v1), VertexHandle(v2), true);
		}
	}

	/*
	* Faces
	*/
	getCleanLine(iff, line);
	sstr.clear();
	sstr.str(line);
	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp != "FACES") {
		std::cerr << "No face section defined!" << std::endl;
		return false;
	}
	else {

		// Read in number of faces
		getCleanLine(iff, line);
		sstr.clear();
		sstr.str(line);
		sstr >> c;

		// Read in faces
		for (uint64_t i = 0u; i < c; ++i) {

			getCleanLine(iff, line);
			sstr.clear();
			sstr.str(line);

			std::vector<HalfEdgeHandle> hes;

			// Get face valence
			uint64_t val = 0u;
			sstr >> val;

			// Read half-edge indices
			for (unsigned int e = 0; e < val; ++e) {

				unsigned int v1 = 0;
				sstr >> v1;
				hes.push_back(HalfEdgeHandle(v1));
			}

			_mesh.add_face(hes, _topologyCheck);
		}
	}

	/*
	* Cells
	*/
	getCleanLine(iff, line);
	sstr.clear();
	sstr.str(line);
	sstr >> s_tmp;
	std::transform(s_tmp.begin(), s_tmp.end(), s_tmp.begin(), ::toupper);
	if (s_tmp != "POLYHEDRA") {
		std::cerr << "No polyhedra section defined!" << std::endl;
		return false;
	}
	else {

		// Read in number of cells
		getCleanLine(iff, line);
		sstr.clear();
		sstr.str(line);
		sstr >> c;

		// Read in cells
		for (uint64_t i = 0u; i < c; ++i) {

			getCleanLine(iff, line);
			sstr.clear();
			sstr.str(line);

			std::vector<HalfFaceHandle> hfs;

			// Get cell valence
			uint64_t val = 0u;
			sstr >> val;

			// Read half-face indices
			for (unsigned int f = 0; f < val; ++f) {

				unsigned int v1 = 0;
				sstr >> v1;
				hfs.push_back(HalfFaceHandle(v1));
			}

			_mesh.add_cell(hfs, _topologyCheck);
		}
	}

	//while (!iff.eof()) {
	//	// "End of file reached while searching for input!"
	//	// is thrown here. \TODO Fix it!

	//	// Read property
	//	readProperty(iff, _mesh);
	//}

	if (_computeBottomUpIncidences) {
		// Compute bottom-up incidences
		_mesh.enable_bottom_up_incidences(true);
	}

	std::cerr << "######## openvolumemesh info #########" << std::endl;
	std::cerr << "#vertices: " << _mesh.n_vertices() << std::endl;
	std::cerr << "#edges:    " << _mesh.n_edges() << std::endl;
	std::cerr << "#faces:    " << _mesh.n_faces() << std::endl;
	std::cerr << "#cells:    " << _mesh.n_cells() << std::endl;
	std::cerr << "######################################" << std::endl;

	return true;
}

//template <class MeshT>
void myWriteFile(const std::string& _filename, TetrahedralMesh &_mesh)
{
	std::ofstream off(_filename.c_str(), std::ios::out);
	// Write header
	off << "OVM ASCII" << std::endl;

	uint64_t n_vertices(_mesh.n_vertices());
	off << "Vertices" << std::endl;
	off << n_vertices << std::endl;

	typedef typename TetrahedralMesh::PointT Point;

	// write vertices
	for (VertexIter v_it = _mesh.v_iter(); v_it; ++v_it) {

		Point v = _mesh.vertex(*v_it);
		off << v[0] << " " << v[1] << " " << v[2] << std::endl;
	}

	uint64_t n_edges(_mesh.n_edges());
	off << "Edges" << std::endl;
	off << n_edges << std::endl;

	// write edges
	for (EdgeIter e_it = _mesh.e_iter(); e_it; ++e_it) {

		VertexHandle from_vertex = _mesh.edge(*e_it).from_vertex();
		VertexHandle to_vertex = _mesh.edge(*e_it).to_vertex();
		off << from_vertex << " " << to_vertex << std::endl;
	}

	uint64_t n_faces(_mesh.n_faces());
	off << "Faces" << std::endl;
	off << n_faces << std::endl;

	// write faces
	for (FaceIter f_it = _mesh.f_iter(); f_it; ++f_it) {

		off << static_cast<uint64_t>(_mesh.face(*f_it).halfedges().size()) << " ";

		std::vector<HalfEdgeHandle> halfedges = _mesh.face(*f_it).halfedges();

		for (typename std::vector<HalfEdgeHandle>::const_iterator it = halfedges.begin(); it
			!= halfedges.end(); ++it) {

			off << it->idx();

			if ((it + 1) != halfedges.end())
				off << " ";
		}

		off << std::endl;
	}

	uint64_t n_cells(_mesh.n_cells());
	off << "Polyhedra" << std::endl;
	off << n_cells << std::endl;

	for (CellIter c_it = _mesh.c_iter(); c_it; ++c_it) {

		off << static_cast<uint64_t>(_mesh.cell(*c_it).halffaces().size()) << " ";

		std::vector<HalfFaceHandle> halffaces = _mesh.cell(*c_it).halffaces();

		for (typename std::vector<HalfFaceHandle>::const_iterator it = halffaces.begin(); it
			!= halffaces.end(); ++it) {

			off << it->idx();

			if ((it + 1) != halffaces.end())
				off << " ";
		}

		off << std::endl;
	}

}