import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from concurrent.futures import process
import numpy as np
import trimesh
import pyrender
import cv2
import jittor as jt
jt.flags.use_cuda = 1
import time, functools

def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = fn(*args, **kwargs)
        print('%s executed in %s s' % (fn.__name__, (time.time() - t1)))
        return res
    return wrapper


def render_depth(c2w, mesh, FOCAL=555.555, IMGSIZE=400, aspect_ratio=1.0) -> np.array:

    scene = pyrender.Scene()

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    camera_pose = c2w
    camera = pyrender.PerspectiveCamera(yfov=2*np.arctan(IMGSIZE/FOCAL/2), aspectRatio=aspect_ratio)

    scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(int(IMGSIZE*aspect_ratio), IMGSIZE)
    _, depth = r.render(scene)
    return depth


def load_objs_as_meshes(mesh_files):
    if isinstance(mesh_files, list):
        res = [load_objs_as_meshes(x) for x in mesh_files]
    else:
        res = trimesh.load_mesh(mesh_files, process=False, maintain_order=True)
    return res


import re
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def readTXT(txt_file:str) -> tuple:
    with open(txt_file, 'r') as f:
        NumV = int(f.readline())
        verts, tets = [],[]
        for _ in range(NumV):
            verts.append(np.fromstring(f.readline(), sep = " "))
        NumT = int(f.readline())
        for _ in range(NumT):
            tets.append(np.fromstring(f.readline(), sep=" ", dtype=np.int32))
    verts = np.stack(verts)
    tets = np.stack(tets)
    verts = jt.array(verts, dtype=jt.float32)
    tets = jt.array(tets, dtype=jt.int32)
    return verts, tets


def readVertsofOVM(ovm_path:str):
    verts = []
    with open(ovm_path, 'r') as vmf:
        lines = vmf.readlines()
    NumV = int(lines[2])
    for idx in range(3, 3+NumV):
        verts.append(np.fromstring(lines[idx], sep=" ", dtype=np.float))
    verts = np.stack(verts, axis=0)
    return jt.array(verts, dtype=jt.float32)


class jt_KNN(jt.nn.Module):
    def __init__(self, k):
        self.k = k
        self.cuda_inc= """
        #undef out
        #include "helper_cuda.h" 
        __global__ void compute_distances(float * ref,
                                        int     ref_width,
                                        int     ref_pitch,
                                        float * query,
                                        int     query_width,
                                        int     query_pitch,
                                        int     height,
                                        float * dist) {
            // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
            const int BLOCK_DIM = 16;
            __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
            __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];
            // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
            __shared__ int begin_A;
            __shared__ int begin_B;
            __shared__ int step_A;
            __shared__ int step_B;
            __shared__ int end_A;
            // Thread index
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int batch_id = blockIdx.z;
            // Initializarion of the SSD for the current thread
            float ssd = 0.f;
            // Loop parameters
            begin_A = BLOCK_DIM * blockIdx.y;
            begin_B = BLOCK_DIM * blockIdx.x;
            step_A  = BLOCK_DIM * ref_pitch;
            step_B  = BLOCK_DIM * query_pitch;
            end_A   = begin_A + (height-1) * ref_pitch;
            // Conditions
            int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
            int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
            int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix
            // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
            for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
                // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
                if (a/ref_pitch + ty < height) {
                    shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx + batch_id * height * ref_pitch] : 0;
                    shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx + batch_id * height * query_pitch] : 0;
                }
                else {
                    shared_A[ty][tx] = 0;
                    shared_B[ty][tx] = 0;
                }
                // Synchronize to make sure the matrices are loaded
                __syncthreads();
                // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
                if (cond2 && cond1) {
                    for (int k = 0; k < BLOCK_DIM; ++k){
                        float tmp = shared_A[k][ty] - shared_B[k][tx];
                        ssd += tmp*tmp;
                    }
                }
                // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
                __syncthreads();
            }
            // Write the block sub-matrix to device memory; each thread writes one element
            if (cond2 && cond1) {
                dist[ (begin_A + ty) * query_pitch + begin_B + tx + batch_id * ref_pitch * query_pitch ] = ssd;
            }
        }
        __global__ void modified_insertion_sort(float * dist,
                                                int     ref_pitch,
                                                int *   index,
                                                int     index_pitch,
                                                int     width,
                                                int     height,
                                                int     k){
            // Column position
            unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
            int batch_id = blockIdx.z ;
            
            // Do nothing if we are out of bounds
            if (xIndex < width) {
                // Pointer shift
                float * p_dist  = dist  + xIndex + batch_id * ref_pitch * index_pitch;
                int *   p_index = index + xIndex + batch_id * index_pitch * k;
                // Initialise the first index
                p_index[0] = 0;
                // Go through all points
                for (int i=1; i<height; ++i) {
                    // Store current distance and associated index
                    float curr_dist = p_dist[i*index_pitch];
                    int   curr_index  = i;
                    // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
                    if (i >= k && curr_dist >= p_dist[(k-1)*index_pitch]) {
                        continue;
                    }
                    // Shift values (and indexes) higher that the current distance to the right
                    int j = min(i, k-1);
                    while (j > 0 && p_dist[(j-1)*index_pitch] > curr_dist) {
                        p_dist[j*index_pitch]   = p_dist[(j-1)*index_pitch];
                        p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                        --j;
                    }
                    // Write the current distance and index at their position
                    p_dist[j*index_pitch]   = curr_dist;
                    p_index[j*index_pitch] = curr_index; 
                }
            }
        }
            __global__ void compute_sqrt(float * dist, int width, int pitch, int k){
                unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
                int batch_id = blockIdx.z;
                if (xIndex<width && yIndex<k)
                    dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
            }
           inline static bool knn_cuda_global(
               int batch_size, 
               float * ref,
                    int           ref_nb,
               float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int *         knn_index, 
                     float *  tmp_dist ){
            // Constants
            const int BLOCK_DIM = 16;
            const unsigned int size_of_float = sizeof(float);
            const unsigned int size_of_int   = sizeof(int);
            // Return variables
            cudaError_t err0, err1, err2, err3;
            // Allocate global memory
            float * ref_dev   = ref;
            float * query_dev = query;
            float * dist_dev  = tmp_dist;
            int   * index_dev = knn_index;
            // Deduce pitch values
            size_t ref_pitch   = ref_nb; 
            size_t query_pitch = query_nb;
            size_t dist_pitch  = query_nb; 
            size_t index_pitch = query_nb; 
            // Compute the squared Euclidean distances
            dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
            dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, batch_size);
            if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
            if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;
            // printf("%d", cudaDeviceSynchronize()); 
            // checkCudaErrors(cudaDeviceSynchronize());
            // printf(" before compute_distances \\n");
            compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
            // checkCudaErrors(cudaDeviceSynchronize());
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after compute_distances \\n");
            // Sort the distances with their respective indexes
            dim3 block1(256, 1, 1);
            dim3 grid1(query_nb / 256, 1, batch_size);
            if (query_nb % 256 != 0) grid1.x += 1;
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" before modified_insertion_sort \\n");
            // checkCudaErrors(cudaDeviceSynchronize());
            modified_insertion_sort<<<grid1, block1>>>(dist_dev, ref_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
            // checkCudaErrors(cudaDeviceSynchronize());
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after modified_insertion_sort \\n");
            
            // Compute the square root of the k smallest distances
            //dim3 block2(16, 16, 1);
            //dim3 grid2(query_nb / 16, k / 16, batch_size);
            //if (query_nb % 16 != 0) grid2.x += 1;
            //if (k % 16 != 0)        grid2.y += 1;
            //compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	
            // Copy k smallest distances / indexes from the device to the host
            // TODO: batch 2d copy dist
            // cudaMemcpy2DAsync(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch*size_of_float,  query_nb * size_of_float, k, cudaMemcpyDefault);
            return true;
        }
        """
        self.cuda_src = '''
            const int k = out0_shape1;
            const int query_nb = in1_shape2; 
            const int ref_nb = in0_shape2;
            const int dim = in0_shape1;
            const int batch_size = in0_shape0;
            knn_cuda_global(batch_size, in0_p, ref_nb, in1_p, query_nb, dim, k, out0_p, in2_p);
        '''

    def execute(self, x_q, x_r): # n_points, c_dim
        x_q, x_r = x_q.transpose().unsqueeze(0), x_r.transpose().unsqueeze(0)
        batch_size, c_dim, q_points = x_q.shape 
        batch_size, c_dim, r_points = x_r.shape 
        out_idx_shapes = [batch_size, self.k, q_points]
        tmp_dist = jt.empty((batch_size, r_points, q_points), "float32")
        idxs,  = jt.code(
            [out_idx_shapes],
            ['int32'],
            [x_r, x_q, tmp_dist], # in0 r point in1 q point 
            cuda_src=self.cuda_src,
            cuda_header=self.cuda_inc,
        )
        return idxs[0].transpose()



from sklearn.neighbors import NearestNeighbors
class TetMesh():
    """
    A class to represent a Tetradral mesh.

    ...

    Attributes
    ----------
    verts : vertices of the mesh, [NV,3]
    tets : tethedrals of the mesh, [NT,4]
    NV : number of verts
    NT: number of tets
    deltas : deformation deltas of each vertices
    indexes: use for calculate det, check which tet the sample point is in

    Methods
    -------
    readTXT(txt_file):
        read verts and tets from .txt file
    findTet(sample_pts):
        query the delta of sample points
    """

    def __init__(self, verts, tets) -> None:
        self.verts, self.tets = verts, tets
        self.tet_verts = self.verts[self.tets] # [NT,4,3]
        self.NV = self.verts.shape[0]
        self.NT = self.tets.shape[0]
        self.genIndex()
        self.vert2tet = self.calVert2Tet() # [NV, max_degree]
        verts = self.verts.reshape(-1,3).numpy()  # [N,3]
        self.nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(verts)
        self.knn_fun = jt_KNN(4)


    def calVert2Tet(self):
        """calculate the tet idx for each verts.
            Padding -1 if tets number is less than max_degree
        """
        vert2tet = [[] for _ in range(self.NV)]
        max_degree = 0
        for idx, tet in enumerate(self.tets):
            for vID in tet:
                vID = int(vID)
                vert2tet[vID].append(idx)
                max_degree = max(max_degree, len(vert2tet[vID]))
        for x in vert2tet:
            while (len(x) < max_degree):
                x.append(-1)
        vert2tet = jt.array(vert2tet).long()
        self.max_degree = max_degree
        print("the max degree of vertices in tets is %d" % self.max_degree)
        return vert2tet

    def genIndex(self) -> None:
        indexes = []
        index = jt.array([[1,2,3,4]]).long().transpose().expand(4,3)
        indexes.append(index.unsqueeze(0))
        index = jt.array([[0,2,3,4]]).long().transpose().expand(4,3)
        indexes.append(index.unsqueeze(0))
        index = jt.array([[1,0,3,4]]).long().transpose().expand(4,3)
        indexes.append(index.unsqueeze(0))
        index = jt.array([[1,2,0,4]]).long().transpose().expand(4,3)
        indexes.append(index.unsqueeze(0))
        index = jt.array([[1,2,3,0]]).long().transpose().expand(4,3)
        indexes.append(index.unsqueeze(0))
        self.indexes = jt.concat(indexes, dim=0) # [5,4,3]

    def findTet(self, sample_pts) -> tuple:
        """[calculate the barycentric coord for each pt and each tet]
        """
        bs, NT = sample_pts.shape[0], self.NT
        tet_verts = self.tet_verts.unsqueeze(0).expand(bs,NT,4,3)
        sample_pts = sample_pts.unsqueeze(1).expand(bs,NT,3)

        return self.findTetCore(sample_pts, tet_verts)

    def findTetCore(self, sample_pts, tet_verts) -> tuple:
        ### need to be accelerated!
        with jt.no_grad():
            bs, NT = sample_pts.shape[:2]
            all_verts = jt.concat([sample_pts.unsqueeze(2),tet_verts], dim=2) # [bs,NT,5,3]
            indexes = self.indexes.unsqueeze(0).expand(bs*NT,5,4,3).reshape(bs,NT,5,4,3)
            all_verts = all_verts.unsqueeze(2).expand(bs,NT,5,5,3) # [bs,NT,5,5,3]

            all_dets = jt.gather(all_verts, dim=3, index=indexes) # [bs,NT,5,4,3]
            all_dets = jt.concat([all_dets, jt.ones(list(all_dets.shape[:-1])+[1])], dim=-1) # [bs,NT,5,4,4]

            all_dets = jt.linalg.det(all_dets) # [bs,NT,5]

            uvwz = all_dets[...,1:] / (all_dets[...,0:1] + 1e-9) # [bs,NT,4]
            u, v, w, z = uvwz[...,0], uvwz[...,1], uvwz[...,2], uvwz[...,3] # [bs,NT]

            in_tet_mask = jt.logical_and(u > 0, v > 0)
            in_tet_mask = jt.logical_and(in_tet_mask, w > 0)
            in_tet_mask = jt.logical_and(in_tet_mask, z > 0) # [bs,NT]

            in_tet_mask = jt.concat([in_tet_mask.int(), jt.ones((bs,1),dtype=jt.int32)], dim=-1) # [bs,NT+1]
            tet_idx = jt.argmax(in_tet_mask, dim=1)[0] # [bs,]

            uvwz = jt.concat([uvwz, jt.zeros([bs,1,4])], dim=1) # [bs,NT+1,4]
            bary_index = tet_idx.unsqueeze(-1).unsqueeze(-1).expand(bs,1,4)
            barycentric = jt.gather(uvwz, dim=1, index=bary_index).permute(0,2,1) # [bs,4,1]

            tet_idx[tet_idx==NT] = -1

        return tet_idx, barycentric

    def findTetKNN(self, sample_pts):
        """find the Tet using KNN. Only find in K tets indicated by the nearest k points
        """
        with jt.no_grad():
            bs, NT = sample_pts.shape[0], self.NT
            #          

            sample_pts = sample_pts.reshape(1,-1,3)  # [1,N,3]
            verts = self.verts.reshape(1,-1,3)  # [1,N,3]
            K = 4 # nearnest points
            # res = knn_points(sample_pts, verts, K=K)
            t1 = time.time()
            # _, verts_id = self.nbrs.kneighbors(sample_pts[0].numpy())  # 
            # verts_id = jt.array(verts_id).long()
            verts_id = self.knn_fun(sample_pts[0], verts[0])
            # print("knn cost time %s" % (time.time() - t1))
            ### construct tets [bs,K*max_degree,4,3] for calculation
            # verts_id = res.idx[0] # [bs, K]
            tets_id = self.vert2tet[verts_id] # [bs, K, max_degree]
            # expand the dimension of tet_verts, to padding the tets
            pad_tet_verts = jt.concat([self.tet_verts, jt.rand((1,4,3),dtype=jt.float32)+999], dim=0) # [NT+1,4,3]
            tets_id[tets_id==-1] = NT
            cur_tet_verts = pad_tet_verts[tets_id.reshape(bs,-1)] # [bs,K*max_degree,4,3]. gather cannot use -1 as index, while slice can！
            cur_NT = K * self.max_degree

            sample_pts = sample_pts.permute(1,0,2).expand(bs,cur_NT,3)

            tets_local_id, barycentric = self.findTetCore(sample_pts, cur_tet_verts) # [bs,], [bs,4,1]
            # tets_local_id[tets_local_id!=-1] = tets_id.reshape(bs,-1)[tets_local_id[tets_local_id!=-1]] # change local to global, too complex ...
            tets_id_ = jt.concat([tets_id.reshape(bs,-1),-1*jt.ones((bs,1), dtype=jt.int32)], dim=1) # [bs,K*max_degree+1]
            tets_local_id[tets_local_id==-1] = cur_NT
            result = jt.gather(tets_id_, dim=1, index=tets_local_id.unsqueeze(-1))

        # return jt.zeros_like(result.squeeze(-1)), jt.zeros_like(barycentric)
        return result.squeeze(-1), barycentric


def genConvexhullVolume(reconstructed_mesh_file:str, deformed_mesh_file:str, fix_camera = False) -> tuple:
    """[summary]

    Args:
        reconstructed_mesh_file (str): [reconstructed file ,txt format]
        deformed_mesh_file (str): [deformaed file, ovm format]
        fix_camera (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: [tri, deltas]
    """
    ori_verts, tets = readTXT(reconstructed_mesh_file)

    if fix_camera:
        import glob, os
        deformed_mesh_files = sorted(glob.glob(os.path.join(deformed_mesh_file, '*.ovm')))
        deformed_verts = [readVertsofOVM(x) for x in deformed_mesh_files]

        tri = [TetMesh(x, tets) for x in deformed_verts]
        deltas = [ori_verts-x for x in deformed_verts]
        print("finish constructing Cage hulls !")
        return tri, deltas
    else:
        deformed_verts = readVertsofOVM(deformed_mesh_file)
        Num = 40
        deltas = ori_verts - deformed_verts

        tri = TetMesh(deformed_verts, tets)
        print("finish constructing Cage hulls !")
        return tri, deltas

def queryDelta(hull, deltas, query_pts):
    '''
        hull: the tet mesh
        deltas: [NV,3]
        query_pts: [bs, N, 3]
    '''
    bs, N, _ = query_pts.shape
    query_pts = query_pts.reshape(-1,3)

    t1 = time.time()
    with jt.no_grad():
        # simplexID, barycentric = hull.findTet(query_pts) # [N,], [N,4,1]
        simplexID, barycentric = hull.findTetKNN(query_pts) # [N,], [N,4,1]
        # simplexID_ = simplexID.numpy()
        # barycentric_ = barycentric.numpy()
        # print("find TNN cost time %s" % (time.time()-t1))

        zero_mask = (simplexID == -1).reshape(bs, N, -1)
        simplexID[simplexID==-1] = hull.NT-1
        values = deltas[hull.tets[simplexID]] # [N,4,3]

        result = (barycentric * values).sum(dim=1)
        # print("the hull has %d simplicies, query %d points with time [%f]" % (len(hull.tets), len(result), time.time()-t1))

        result = result.reshape(bs, N, -1)
        tri_verts = hull.verts[hull.tets[simplexID]].reshape(bs, N, 4, 3)
        tri_deltas = values.reshape(bs, N, 4, 3)


    return result, (tri_verts,tri_deltas,zero_mask)

