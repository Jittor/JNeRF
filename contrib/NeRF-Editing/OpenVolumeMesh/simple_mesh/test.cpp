// C++ includes
#include <iostream>
#include <vector>

// Include vector classes
#include <OpenVolumeMesh/Geometry/VectorT.hh>

// Include polyhedral mesh kernel
#include <OpenVolumeMesh/Mesh/PolyhedralMesh.hh>

// Include the file manager header
#include <OpenVolumeMesh/FileManager/FileManager.hh>

#include <math.h>  // sqrt

// Make some typedefs to facilitate your life
typedef OpenVolumeMesh::Geometry::Vec3f         Vec3f;
typedef OpenVolumeMesh::GeometryKernel<Vec3f>   PolyhedralMeshV3f;

float calDihedralAngle(std::vector<OpenVolumeMesh::Geometry::Vec3f> dihedral) {
    // v1, v2 constructs the common edge. v3 and v4 expand the dihedral angle
    OpenVolumeMesh::Geometry::Vec3f v1 = dihedral[0];
    OpenVolumeMesh::Geometry::Vec3f v2 = dihedral[1];
    OpenVolumeMesh::Geometry::Vec3f v3 = dihedral[2];
    OpenVolumeMesh::Geometry::Vec3f v4 = dihedral[3];

    // assume the order: v3 -> v1 -> v2 -> v4
    OpenVolumeMesh::Geometry::Vec3f d1 = v1 - v3;
    OpenVolumeMesh::Geometry::Vec3f d2 = v2 - v1;
    OpenVolumeMesh::Geometry::Vec3f d3 = v4 - v2;

    OpenVolumeMesh::Geometry::Vec3f n1 = d1.cross(d2);
    OpenVolumeMesh::Geometry::Vec3f n2 = d2.cross(d3);

    float cos_angle = n1.dot(n2) / n1.norm() / n2.norm();

    return cos_angle / sqrt(1 - cos_angle*cos_angle);
}

int main(int argc, char const *argv[])
{
    std::string ovm_path = "/mnt/2/sunyangtian/NeRF_Ali/volumeARAP_batch/test_file2/mesh_cage_.ovm";
    OpenVolumeMesh::GeometricPolyhedralMeshV3f myMesh;
    OpenVolumeMesh::IO::FileManager fileManager;
    fileManager.readFile(ovm_path, myMesh);

    for (int i = 0; i < myMesh.n_vertices(); i++) {
        OpenVolumeMesh::VertexHandle vh(i);
        for (OpenVolumeMesh::VertexVertexIter vvit=myMesh.vv_iter(vh); vvit.valid(); vvit++) {
            OpenVolumeMesh::Geometry::Vec3f pos = myMesh.vertex(*vvit);
            std::cout << "vertex index: " << (*vvit).idx() << std::endl;
        }
        std::cout << "---------------------------------" << std::endl;
        for (OpenVolumeMesh::VertexOHalfEdgeIter voheit=myMesh.voh_iter(vh); voheit.valid(); voheit++) {
            OpenVolumeMesh::VertexHandle vh2 = myMesh.to_vertex_handle(*voheit);
            std::cout << "****** the edge between ****** " << vh.idx() << " and " << vh2.idx() << std::endl;
            OpenVolumeMesh::EdgeHandle eh = myMesh.edge_handle(*voheit);
            for (OpenVolumeMesh::EdgeCellIter ecit=myMesh.ec_iter(eh); ecit.valid(); ecit++) {
                std::cout << "iterate cells ..." << std::endl;
                std::vector<OpenVolumeMesh::Geometry::Vec3f> dihedralAngle;
                // vector size is 4. The first and second is the common edge
                dihedralAngle.push_back(myMesh.vertex(vh));
                dihedralAngle.push_back(myMesh.vertex(vh2));
                std::cout << "specific verts are: " << vh.idx() << ", " << vh2.idx() << std::endl;
                for (OpenVolumeMesh::CellVertexIter cvit=myMesh.cv_iter(*ecit); cvit.valid(); cvit++) {
                    if (*cvit != vh && *cvit != vh2) {
                        dihedralAngle.push_back(myMesh.vertex(*cvit));
                        std::cout << "verts add " << (*cvit).idx() << std::endl;
                    }
                }
                std::cout << "the cot of dihedralAngle is: " << calDihedralAngle(dihedralAngle) << std::endl;
                std::cout << "---------------------------------" << std::endl;
            }

        }
        if (i > 1)
        break;
    }

    std::cout << "TEST dihetral angle" << std::endl;
    OpenVolumeMesh::Geometry::Vec3f v1(0,0,0);
    OpenVolumeMesh::Geometry::Vec3f v2(1,0,0);
    OpenVolumeMesh::Geometry::Vec3f v3(0,0,1);
    OpenVolumeMesh::Geometry::Vec3f v4(0,1,-1);

    std::vector<OpenVolumeMesh::Geometry::Vec3f> dihedral = {v1,v2,v3,v4};
    std::cout << "the cot of dihedralAngle is: " << calDihedralAngle(dihedral) << std::endl;

    return 0;
}
