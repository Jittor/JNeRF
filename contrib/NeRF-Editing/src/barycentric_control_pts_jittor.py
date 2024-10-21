"""input: mesh file, tetrahedral txt file and deformed mesh file. 
    Calculate the tet ID of each vertices and its barycentric coordinate.
   output: the control points for each vertices.
"""
import trimesh
import jittor as jt
from utils import TetMesh, readTXT
import numpy as np
import glob

def main(mesh_path, tet_path, deformed_path, check_output=False):
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
    mesh_verts = jt.array(np.asarray(mesh.vertices)).float()
    tet_verts, tet_idx = readTXT(tet_path)

    tet_mesh = TetMesh(tet_verts, tet_idx)

    tet_ids, barys = batchify(mesh_verts, tet_mesh)

    save_path = deformed_path.replace(".obj", "_barycentric_control_simple3.txt")
    deformed_verts = trimesh.load_mesh(deformed_path, process=False, maintain_order=True).vertices
    deformed_verts = jt.array(np.asarray(deformed_verts)).float()

    ### add check output module
    if check_output:
        tet_ids = jt.stack(tet_ids, dim=0)
        barys = jt.stack(barys, dim=0) # [N,4,1]
        assert len(mesh_verts) == len(tet_ids)
        values = tet_mesh.verts[tet_ids] # [N,4,3]
        verts_new = (values * barys).sum(dim=1)
        print(verts_new == mesh_verts)
        import ipdb; ipdb.set_trace()
    try:
        saveControlPts(tet_ids, barys, deformed_verts, save_path)
        
    except:
        import ipdb; ipdb.set_trace()

def main_seq(mesh_path, tet_path, deformed_paths, check_output=False):
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
    mesh_verts = jt.array(np.asarray(mesh.vertices)).float()
    tet_verts, tet_idx = readTXT(tet_path)

    tet_mesh = TetMesh(tet_verts, tet_idx)

    tet_ids, barys = batchify(mesh_verts, tet_mesh, chunk=100)

    save_path = deformed_paths[0].replace(".obj", "_barycentric_control.txt")
    def f(x):
        out = trimesh.load_mesh(x, process=False, maintain_order=True).vertices
        out = jt.array(np.asarray(out)).float()
        return out
    deformed_verts = list(map(f, deformed_paths))

    saveControlPtsSeq(tet_ids, barys, deformed_verts, save_path)

def batchify(verts, tet_mesh, chunk=100):
    """find verts in which tets in batch.
    """
    tet_ids, barys = [], []
    for i in range(0, verts.shape[0], chunk):
        print("quering one chunk ...")
        tet_id, barycentric = tet_mesh.findTet(verts[i:i+chunk])
        tet_ids += [tet_mesh.tets[x] for x in tet_id]
        barys += [x for x in barycentric]
    return tet_ids, barys  # list: [[4,]...], [[4,1]...]


def saveControlPts(tet_ids, barys, deformed_verts, control_txt_path) -> None:
    with open(control_txt_path, 'w') as cf:
        cf.write('1\n')
        cf.write('%d\n' % len(deformed_verts))
        print("saving %d control points coordinate" % (len(deformed_verts)))
        for vert in deformed_verts:
            cf.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
        cf.write('%d\n' % len(tet_ids))
        print("saving %d tet verts idx and barycentric coordinate" % (len(tet_ids)))
        for tet_id, bary in zip(tet_ids, barys):
            cf.write("%d %d %d %d\n" % (tet_id[0], tet_id[1], tet_id[2], tet_id[3]))
            cf.write("%f %f %f %f\n" % (bary[0], bary[1], bary[2], bary[3]))
    print("write control txt to %s" % control_txt_path)

def saveControlPtsSeq(tet_ids, barys, deformed_verts, control_txt_path) -> None:
    with open(control_txt_path, 'w') as cf:
        cf.write('%d\n' % (len(deformed_verts)))  # number of seqence
        print("saving %d sequence" % (len(deformed_verts)))
        for idx, dv in enumerate(deformed_verts):
            cf.write('%d\n' % len(dv))
            print("saving %d item, with %d control points coordinate" % (idx, len(dv)))
            for vert in dv:
                cf.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
        cf.write('%d\n' % len(tet_ids))
        print("saving %d tet verts idx and barycentric coordinate" % (len(tet_ids)))
        for tet_id, bary in zip(tet_ids, barys):
            tet_id = tet_id[0]
            cf.write("%d %d %d %d\n" % (tet_id[0], tet_id[1], tet_id[2], tet_id[3]))
            cf.write("%f %f %f %f\n" % (bary[0], bary[1], bary[2], bary[3]))
    print("write control txt to %s" % control_txt_path)


if __name__ == "__main__":
    mesh_path = "./logs/hbychair_wo_mask/mesh_nofloor_simp.obj"
    tet_path = "./logs/hbychair_wo_mask/mesh_cage_nofloor_.txt"
    deformed_dir = "./logs/hbychair_wo_mask/mesh_seq/*.obj"

    deformed_paths = sorted(glob.glob(deformed_dir))[-1:]
    main_seq(mesh_path, tet_path, deformed_paths)