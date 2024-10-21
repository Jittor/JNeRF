import trimesh
import numpy as np
import open3d as o3d
from trainers.utils.vis_utils import imf2mesh


def trimesh_to_o3dmesh(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.array(mesh.faces))
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def o3dmesh_to_trimesh(mesh):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).reshape(-1, 3).astype(np.float),
        faces=np.asarray(mesh.triangles).reshape(-1, 3).astype(np.int)
    )
    return mesh


def deform_mesh_o3d(imf, handles, targets, normalize=True, res=256,
                    imf_mesh=None, steps=50, smoothed_alpha=0.01, verbose=True):
    """
    Use Open3D to do deformation
    Args:
        [imf]
        [handles] (n, 3) Source points.
        [targets] (n, 3) Target points.
        [normalize] Whether normalize the mesh to unit sphere. Default (True).
        [res] Resolution for MC. Default (256).
    Returns:
    """
    if imf_mesh is None:
        mesh = imf2mesh(imf, res=res, threshold=0.00)

        if normalize:
            verts = (mesh.vertices * 2 - res) / float(res)
            mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)
    else:
        mesh = imf_mesh

    vertices = np.asarray(mesh.vertices).reshape(-1, 3)
    vert_ids = []
    vert_pos = []
    for i in range(handles.reshape(-1, 3).shape[0]):
        dist = np.linalg.norm(
            vertices - handles[i, :].reshape(1, 3), axis=-1
        ).flatten()
        handle_idx = np.argmin(dist)
        vert_ids.append(handle_idx)
        vert_pos.append(
            vertices[handle_idx].reshape(3) + targets[i].reshape(3) -
            handles[i].reshape(3))

    constraint_ids = o3d.utility.IntVector(vert_ids)
    constraint_pos = o3d.utility.Vector3dVector(vert_pos)
    o3d_vert0 = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_face0 = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh0 = o3d.geometry.TriangleMesh(
        vertices=o3d_vert0, triangles=o3d_face0)
    o3d_mesh0.compute_vertex_normals()

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        if smoothed_alpha > 0:
            print("Smoothing alphas:", smoothed_alpha, "Use Smoothed Energy")
            mesh_deformed = o3d_mesh0.deform_as_rigid_as_possible(
                constraint_ids, constraint_pos, max_iter=steps,
                smoothed_alpha=smoothed_alpha,
                energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed)
        else:
            print("Smoothing alphas:", smoothed_alpha, "Use Spokes Energy")
            mesh_deformed = o3d_mesh0.deform_as_rigid_as_possible(
                constraint_ids, constraint_pos, max_iter=steps,
                smoothed_alpha=0,
                energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes)

    return o3dmesh_to_trimesh(mesh_deformed)


