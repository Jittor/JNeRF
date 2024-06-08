// C++ includes
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

// Include vector classes
#include <OpenVolumeMesh/Geometry/VectorT.hh>

// Include polyhedral mesh kernel
#include <OpenVolumeMesh/Mesh/PolyhedralMesh.hh>

#include <OpenVolumeMesh/FileManager/FileManager.hh>

// Make some typedefs to facilitate your life
typedef OpenVolumeMesh::Geometry::Vec3f         Vec3f;
typedef OpenVolumeMesh::GeometryKernel<Vec3f>   PolyhedralMeshV3f;

// verts [NV,3]; simplicies [NC,4]
void readVerts(std::string file_path, std::vector<std::vector<float>>& verts, std::vector<std::vector<int>>& cells) {
  std::ifstream myfile (file_path);
  if (myfile.is_open()) {
    int Num;
    myfile >> Num; // read verts
    for (int idx = 0; idx < Num; idx++) {
        float v1, v2, v3; myfile >> v1; myfile>> v2; myfile >> v3;
        std::vector<float> cur = {v1, v2, v3};
        verts.push_back(cur);
    }
    myfile >> Num; // read cells
    for (int idx = 0; idx < Num; idx++) {
        int id1, id2, id3, id4; 
        myfile >> id1; myfile >> id2; myfile >> id3; myfile >> id4;
        std::vector<int> cur = {id1, id2, id3, id4};
        cells.push_back(cur);
    }
        myfile.close();
    }

  else std::cout << "Unable to open file \n"; 
}

int getHalfFaceID(std::vector<Vec3f>& vertsPerCell) {
    // determine the face[v1/v2/v3] ID wrt. v4
    Vec3f v01 = vertsPerCell[1] - vertsPerCell[0];
    Vec3f v02 = vertsPerCell[2] - vertsPerCell[0];
    Vec3f v03 = vertsPerCell[3] - vertsPerCell[0];

    // for (Vec3f v : vertsPerCell) std::cout << v << ", ";
    // std::cout << std::endl;

    if (v01.cross(v02).dot(v03) > 0) return 0;
    else return 1;
}
 
int main(int _argc, char** _argv) {


    std::string input_file = _argv[1];
    std::string save_name = _argv[2];
    std::vector<std::vector<float>> verts;
    std::vector<std::vector<int>> cells;
    readVerts(input_file, verts, cells);

    // Create mesh object
    PolyhedralMeshV3f myMesh;

    // add verts
    std::vector<OpenVolumeMesh::VertexHandle> vert_handles;
    for (std::vector<float> cur : verts) {
        OpenVolumeMesh::VertexHandle vh = myMesh.add_vertex(Vec3f(cur[0], cur[1], cur[2]));
        vert_handles.push_back(vh);
    }

    // add faces
    // to avoid duplicate faces, use hash to check
    std::unordered_map<std::string, OpenVolumeMesh::FaceHandle> fh_hash;
    std::vector<OpenVolumeMesh::FaceHandle> face_handles;
    for (std::vector<int> cur_cell : cells) {
        std::vector<OpenVolumeMesh::VertexHandle> cur_vh;
        OpenVolumeMesh::FaceHandle fh;
        std::vector<OpenVolumeMesh::HalfFaceHandle> halffaces;
        std::vector<Vec3f> vertsPerCell;
        std::string fh_str;

        cur_vh.push_back(vert_handles[cur_cell[0]]); // 0 1 2
        cur_vh.push_back(vert_handles[cur_cell[1]]);
        cur_vh.push_back(vert_handles[cur_cell[2]]);
        fh_str = std::to_string(cur_cell[0]) + ", " + std::to_string(cur_cell[1]) + ", " + std::to_string(cur_cell[2]);
        if (fh_hash.count(fh_str)) fh = fh_hash[fh_str];
        else {fh = myMesh.add_face(cur_vh); fh_hash[fh_str] = fh;}
        for (OpenVolumeMesh::FaceVertexIter fv_it = myMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
            vertsPerCell.push_back(myMesh.vertex(*fv_it));
        }
        vertsPerCell.push_back(myMesh.vertex(vert_handles[cur_cell[3]]));
        halffaces.push_back(myMesh.halfface_handle(fh, getHalfFaceID(vertsPerCell)));
        
        cur_vh.clear(); vertsPerCell.clear();
        cur_vh.push_back(vert_handles[cur_cell[0]]); // 0 1 3
        cur_vh.push_back(vert_handles[cur_cell[1]]);
        cur_vh.push_back(vert_handles[cur_cell[3]]);
        fh_str = std::to_string(cur_cell[0]) + ", " + std::to_string(cur_cell[1]) + ", " + std::to_string(cur_cell[3]);
        if (fh_hash.count(fh_str)) fh = fh_hash[fh_str];
        else {fh = myMesh.add_face(cur_vh); fh_hash[fh_str] = fh;}
        for (OpenVolumeMesh::FaceVertexIter fv_it = myMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
            vertsPerCell.push_back(myMesh.vertex(*fv_it));
        }
        vertsPerCell.push_back(myMesh.vertex(vert_handles[cur_cell[2]]));
        halffaces.push_back(myMesh.halfface_handle(fh, getHalfFaceID(vertsPerCell)));

        cur_vh.clear(); vertsPerCell.clear();
        cur_vh.push_back(vert_handles[cur_cell[0]]); // 0 2 3
        cur_vh.push_back(vert_handles[cur_cell[2]]);
        cur_vh.push_back(vert_handles[cur_cell[3]]);
        fh_str = std::to_string(cur_cell[0]) + ", " + std::to_string(cur_cell[2]) + ", " + std::to_string(cur_cell[3]);
        if (fh_hash.count(fh_str)) fh = fh_hash[fh_str];
        else {fh = myMesh.add_face(cur_vh); fh_hash[fh_str] = fh;}
        for (OpenVolumeMesh::FaceVertexIter fv_it = myMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
            vertsPerCell.push_back(myMesh.vertex(*fv_it));
        }
        vertsPerCell.push_back(myMesh.vertex(vert_handles[cur_cell[1]]));
        halffaces.push_back(myMesh.halfface_handle(fh, getHalfFaceID(vertsPerCell)));

        cur_vh.clear(); vertsPerCell.clear();
        cur_vh.push_back(vert_handles[cur_cell[1]]); // 1 2 3
        cur_vh.push_back(vert_handles[cur_cell[2]]);
        cur_vh.push_back(vert_handles[cur_cell[3]]);
        fh_str = std::to_string(cur_cell[1]) + ", " + std::to_string(cur_cell[2]) + ", " + std::to_string(cur_cell[3]);
        if (fh_hash.count(fh_str)) fh = fh_hash[fh_str];
        else {fh = myMesh.add_face(cur_vh); fh_hash[fh_str] = fh;}
        for (OpenVolumeMesh::FaceVertexIter fv_it = myMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
            vertsPerCell.push_back(myMesh.vertex(*fv_it));
        }
        vertsPerCell.push_back(myMesh.vertex(vert_handles[cur_cell[0]]));
        halffaces.push_back(myMesh.halfface_handle(fh, getHalfFaceID(vertsPerCell)));

        myMesh.add_cell(halffaces);
    }

    // Create file manager object
    OpenVolumeMesh::IO::FileManager fileManager;
    // Store mesh to file "myMesh.ovm" in the current directory
    // std::string save_name = "myMesh_debug.ovm";
    fileManager.writeFile(save_name.c_str(), myMesh);
    std::cout << "save mesh to " << save_name << std::endl;

    return 0;
}
