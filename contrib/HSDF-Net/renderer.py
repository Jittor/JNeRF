import math
import numpy as np
import trimesh
import cv2
import os

import configs.config_loader as cfg_loader

import NDF_combine as NDF

def ray_trace_mesh(trimesh_mesh, xyz_world, rays, batch_size=20004):
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh_mesh)
    all_loc_intersect = []
    all_ray_index = []
    all_face_index = []

    for x in range(0, rays.shape[0], batch_size):
        loc_intersect, ray_index, face_index = \
            intersector.intersects_location(xyz_world[x:x+batch_size], rays[x:x+batch_size])

        if len(ray_index) != 0:
            ray_index = batch_size * int(x / batch_size) + ray_index
            all_loc_intersect.append(loc_intersect)
            all_ray_index.append(ray_index)
            all_face_index.append(face_index)

    if rays.shape[0] % batch_size:
        x = batch_size * int(rays.shape[0] / batch_size)
        loc_intersect, ray_index, face_index = \
            intersector.intersects_location(xyz_world[x:], rays[x:])
        if len(ray_index) != 0:
            ray_index = x + ray_index
            all_loc_intersect.append(loc_intersect)
            all_ray_index.append(ray_index)
            all_face_index.append(face_index)
    
    dists = np.ones(rays.shape[0]) * 100
    normals = np.ones(rays.shape) * 0.1
    
    try:
        all_loc_intersect = np.concatenate(all_loc_intersect, 0)
        all_face_index = np.concatenate(all_face_index, 0)
        all_ray_index = np.concatenate(all_ray_index, 0)

        for ct, xx in enumerate(all_ray_index):
            dist = np.sqrt(np.sum((xyz_world[xx] - all_loc_intersect[ct])**2, 0))

            if dist < dists[xx]:
                dists[xx] = dist
                yy = all_face_index[ct]
                normals_ = trimesh_mesh.face_normals[yy]
                #reverse = np.sum(normals_ * rays[xx]) > 0
                normals__ = -normals_
                #if reverse:
                #    normals__ = -normals_ 
                normals[xx] = normals__
    except:
        dists[dists==100] = -1
        return normals, dists, xyz_world

    
    return normals, dists, all_loc_intersect


def str2bool(inp):
    return inp.lower() in 'true'

class Renderer():
    def __init__(self):
        self.get_args()
        self.create_plane_points_from_bounds()
        self.define_screen_points()
        self.define_unit_rays()

    def get_args(self):
        """
        :return:
        """
        self.args = cfg_loader.get_config()

        # print(self.args.cam_position)
        # print(self.args.cam_orientation)
        os.makedirs(self.args.folder, exist_ok=True)

    def create_plane_points_from_bounds(self):
        """
            Creates a plane of points which acts as the screen for rendering
        """
        # create an xy plane
        x = np.linspace(-self.args.screen_bound, self.args.screen_bound, self.args.size)
        y = np.linspace(-self.args.screen_bound, self.args.screen_bound, self.args.size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))

        # append the third dimension coordinate to the xy plane
        points_list = np.column_stack((X, Y))
        points_list = np.insert(points_list, 2, self.args.screen_depth, axis=1)
        self.points_list = points_list

    def to_rotation_matrix(self):
        """
            Creates rotation matrix from the input euler angles
        """
        euler_angles = np.array(self.args.cam_orientation)
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(math.radians(euler_angles[0])), -math.sin(math.radians(euler_angles[0]))],
                        [0, math.sin(math.radians(euler_angles[0])), math.cos(math.radians(euler_angles[0]))]
                        ])

        R_y = np.array([[math.cos(math.radians(euler_angles[1])), 0, math.sin(math.radians(euler_angles[1]))],
                        [0, 1, 0],
                        [-math.sin(math.radians(euler_angles[1])), 0, math.cos(math.radians(euler_angles[1]))]
                        ])

        R_z = np.array([[math.cos(math.radians(euler_angles[2])), -math.sin(math.radians(euler_angles[2])), 0],
                        [math.sin(math.radians(euler_angles[2])), math.cos(math.radians(euler_angles[2])), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        self.rot_matrix = R

    def to_transf_matrix(self):
        """
            Creates a transformation matrix from rotation matrix and translation vector
        """
        self.to_rotation_matrix()

        temp_trans = np.array([0, 0, 0])
        temp_trans = np.reshape(temp_trans, (1, 3))
        rot = np.concatenate((self.rot_matrix, temp_trans), axis=0)
        rot = np.concatenate((rot, np.reshape(np.array([0, 0, 0, 1]), (4, 1))), axis=1)

        inp_trans = np.reshape(self.args.cam_position, (3,))
        inp_trans = np.concatenate((inp_trans, [1]), axis=0)

        rot[:, 3] = inp_trans

        self.trans_mat = rot

    def append_one(self, arr):
        """
        :param arr:
        :return:
        """
        append = np.ones(arr.shape[0])
        append = np.reshape(append, (append.shape[0], 1))
        new_arr = np.concatenate((arr, append), axis=1)
        return new_arr

    def define_screen_points(self):
        """
            Transforms the screen points and camera position using the camera translation and orientation information provided by the user
        """
        self.create_plane_points_from_bounds()
        self.to_transf_matrix()

        cam_loc = np.array([0, 0, 0])
        screen_and_cam = np.vstack((cam_loc, self.points_list))
        screen_and_cam_hom = self.append_one(screen_and_cam)

        # 4 X SIZE^2
        screen_and_cam_hom_T = np.transpose(screen_and_cam_hom, (1, 0))
        screen_and_cam_hom_T_transformed = np.matmul(self.trans_mat, screen_and_cam_hom_T)

        # SIZE^2 X 4
        screen_and_cam_hom_transformed = np.transpose(screen_and_cam_hom_T_transformed, (1, 0))

        # SIZE^2 X 3
        self.screen_and_cam_transformed = screen_and_cam_hom_transformed[:, :3]

        if self.args.debug_mode:
            trimesh.Trimesh(vertices=self.screen_and_cam_transformed, faces=[]).export('setup_camera_rot.off')

    def define_unit_rays(self):
        """
        Defines rays from camera to the screen along which
        """
        # Separate screen points and camera point
        points = self.screen_and_cam_transformed[1:, :]
        self.cam_trans = np.reshape(self.screen_and_cam_transformed[0, :], (1, 3))

        # Define ray paths from camera
        ray_vector = (points - self.cam_trans)

        # Normalize ray vectors
        norm_ray = np.linalg.norm(ray_vector, ord=2, axis=1)
        norm_ray = np.reshape(norm_ray, (self.args.size * self.args.size, 1))

        self.unit_rays = ray_vector / norm_ray

    def get_lgth_rays(self):
        """
        :return:
        """
        src_batch = np.repeat([self.args.light_position], self.args.size * self.args.size, axis=0)
        rays = src_batch - self.final_points
        norm_ray = np.linalg.norm(rays, ord=2, axis=1)
        norm_ray = np.reshape(norm_ray, (self.args.size * self.args.size, 1))

        self.ray_to_src = rays / norm_ray

    def run(self):
        """
        Runs the ray marching algorithm
        """
        print(self.args)
        path = NDF.loadNDF(
                    mode = 'test', index = self.args.index,
                    pointcloud_samples = self.args.pc_samples,
                    exp_name = self.args.exp_name, data_dir = self.args.data_dir,
                    split_file = self.args.split_file, sample_distribution = self.args.sample_ratio,
                    sample_sigmas = self.args.sample_std_dev, res = self.args.input_res
                    )

        depth = np.zeros((self.args.size * self.args.size, 1))
        depth_gt = np.zeros((self.args.size * self.args.size, 1))

        cam_batch = np.repeat(self.cam_trans, self.args.size * self.args.size, axis=0)
        points = cam_batch.copy()
        points_gt = cam_batch.copy()

        iter = 1

        ray = self.unit_rays.copy()
        ray_gt = self.unit_rays.copy()

        indices_cont_all = list(range(self.args.size * self.args.size))
        indices_cont_all_gt = list(range(self.args.size * self.args.size))

        trimesh_mesh = trimesh.load(os.path.join(path, 'model_scaled.off'))
        # compute gt surface points

        #dists_gt = NDF.predictGtNDF(points, trimesh_mesh)
        #points_gt = points + ray * np.expand_dims(dists_gt, axis=-1)

        '''
        while len(indices_cont_all) > 0:

            print('Iter:', iter)
            dists_points_gt = NDF.predictGtNDF(points_gt, trimesh_mesh)
            dists_points_gt = np.reshape(dists_points_gt, (self.args.size * self.args.size, 1))

            indices_stop_gt = np.where(dists_points_gt < self.args.epsilon)[0]
            indices_stop2_gt = np.where(depth_gt > self.args.max_depth)[0]
            indices_stop_all_gt = list(set(indices_stop_gt).union(set(indices_stop2_gt)))
            # print(len(indices_stop_all))

            ray_gt[indices_stop_all_gt] = 0
            setA = set(range(self.args.size * self.args.size))
            setB = set(indices_stop_all_gt)
            indices_cont_all_gt = list(setA.difference(setB))


            # print(len(indices_cont_all))
            depth[indices_cont_all_gt] = depth[indices_cont_all_gt] + self.args.alpha * dists_points_gt[indices_cont_all_gt]
            points_gt = points_gt + (ray_gt * (self.args.alpha * dists_points_gt))
            iter = iter + 1

        iter = 1
        '''
        
        #print(points.shape)


        while len(indices_cont_all) > 0:

            print('Iter:', iter)
            dists_points = NDF.predictRotNDF(points)
            dists_points = np.reshape(dists_points, (self.args.size * self.args.size, 1))

            indices_stop = np.where(dists_points < self.args.epsilon)[0]
            indices_stop2 = np.where(depth > self.args.max_depth)[0]
            indices_stop_all = list(set(indices_stop).union(set(indices_stop2)))
            # print(len(indices_stop_all))

            ray[indices_stop_all] = 0
            setA = set(range(self.args.size * self.args.size))
            setB = set(indices_stop_all)
            indices_cont_all = list(setA.difference(setB))


            # print(len(indices_cont_all))
            depth[indices_cont_all] = depth[indices_cont_all] + self.args.alpha * dists_points[indices_cont_all]
            points = points + (ray * (self.args.alpha * dists_points))
            iter = iter + 1

        #print(np.max(dists_points))
        #print(points.shape)
        #print(np.max(depth))
        #print(np.min(depth))
        #print(len(depth>self.args.max_depth))

        points = points - (self.unit_rays * self.args.step_back)

        self.final_points = points.copy()

        ## NORMALS
        self.depth_np = depth.copy()
        self.depth_np[self.depth_np > self.args.max_depth] = self.args.max_depth

        dists, gradients = NDF.predictRotGradientNDF(points)

        #dist_gt, gradients_gt = NDF.predictGtGradientNDF(points_gt, trimesh_mesh)
        vertices = trimesh_mesh.vertices.copy()
        trimesh_mesh.vertices[:,1], trimesh_mesh.vertices[:,2] = vertices[:,2], vertices[:,1]
        gradients_gt, depth_gt, all_loc_intersect = ray_trace_mesh(trimesh_mesh, points_gt, ray_gt)

        self.depth_gt = depth_gt.copy()
        self.depth_gt[self.depth_gt > self.args.max_depth] = self.args.max_depth

        self.final_gradients = gradients.copy()
        self.normals = np.reshape(gradients, (self.args.size * self.args.size, 3))

        self.final_gradients_gt = gradients_gt.copy()
        self.normals_gt = np.reshape(gradients_gt, (self.args.size * self.args.size, 3))


    def save(self, image, name, size, normalize):
        """
        :param image: Input image as np array
        :param name: Name of file to be stored
        :param size: Size of the image
        :param normalize: whether to normalize all values to 0-1
        Saves individual images
        """
        if normalize:
            image = (image + 1)/2
        image = np.reshape(image, (self.args.size, self.args.size, size))

        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
        image = image[90:610, :]

        cv2.imwrite(os.path.join(self.args.folder, name), np.uint8(255 * image))

    def save_images(self):
        """
        Saves Images after completion of the rendering algorithm
        """
        shade = np.sum(np.multiply(-self.unit_rays, self.normals), axis=1)
        shade = np.reshape(shade, (shade.shape[0], 1))

        shade[self.depth_np == self.args.max_depth] = 1
        self.save(shade, 'shade.jpg', 1, True)

        # SHADE WITH LIGhT SOURCE
        if self.args.shade:
            self.get_lgth_rays()
            shd_lgth = np.sum(np.multiply(self.ray_to_src, self.normals), axis=1)
            shd_lgth = np.reshape(shd_lgth, (shd_lgth.shape[0], 1))
            shd_lgth[self.depth_np == self.args.max_depth ] = 1
            self.save(shd_lgth, 'shade_src.jpg', 1, True)

        if self.args.normal:
            RGB_normals = self.final_gradients.copy()
            inds = (self.depth_np == self.args.max_depth)
            for j in range(3):
                new_arr = np.reshape(RGB_normals[:, j], (self.args.size * self.args.size, 1))
                new_arr[inds] = 1

            black_pixels_mask = np.all(RGB_normals == [0, 0, 0], axis=-1)
            RGB_normals[black_pixels_mask] = np.array([1, 1, 1])
            self.save(RGB_normals, 'normals.jpg', 3, True)

            # save gt mesh normal map
            RGB_normals_gt = self.final_gradients_gt.copy()
            inds = (self.depth_gt == self.args.max_depth)
            for j in range(3):
                new_arr = np.reshape(RGB_normals_gt[:, j], (self.args.size * self.args.size, 1))
                new_arr[inds] = 1

            black_pixels_mask_gt = np.all(RGB_normals_gt == [0, 0, 0], axis=-1)
            RGB_normals_gt[black_pixels_mask_gt] = np.array([1, 1, 1])
            self.save(RGB_normals_gt, 'normals_gt.jpg', 3, True)

        if self.args.depth:
            depth_normalized = np.copy(self.depth_np / self.args.max_depth)
            self.save(depth_normalized, 'depth_final.jpg', 1, False)

if __name__ == "__main__":
    renderer = Renderer()
    renderer.run()
    renderer.save_images()