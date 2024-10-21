import glob, os, sys
from natsort import natsorted
import numpy as np
from PIL import Image

def gen_data_from_colmap(img_dir, save_dir):
    import colmap_read_model as read_model

    # RGB, pose, intrinsic
    img_list = glob.glob(os.path.join(img_dir, 'images', '*.png'))
    img_list = natsorted(img_list)
    if not os.path.exists(os.path.join(save_dir, 'rgb')):
        os.makedirs(os.path.join(save_dir, 'rgb'), exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, 'pose')):
        os.makedirs(os.path.join(save_dir, 'pose'), exist_ok=True)

    imagesfile = os.path.join(img_dir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    names = [imdata[k].name for k in imdata]
    ids = [imdata[k].id for k in imdata]

    # real_w, real_h = 480, 360
    real_w, real_h = 540, 960

    test_id = []

    for i in range(len(img_list)):
        img_file = img_list[i]
        basename = os.path.basename(img_file)
        if basename not in names:
            continue

        rgb_img = Image.open(img_file).convert('RGB')
        rgb_img = rgb_img.resize((real_w, real_h))

        if i in test_id:
            rgb_img.save(os.path.join(save_dir, 'rgb', '1_test_%04d.png'%(i)))
        else:
            rgb_img.save(os.path.join(save_dir, 'rgb', '0_train_%04d.png'%(i)))

    camerasfile = os.path.join(img_dir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]

    K_savefile = open(os.path.join(save_dir, 'intrinsics.txt'), 'w')
    K_savefile.write('%f %f %f 0.\n' % (f/4, cam.params[1]/4, cam.params[2]/4))
    K_savefile.write('0. 0. 0.\n0.\n1.\n')
    K_savefile.write('%d %d\n' % (real_h, real_w))
    K_savefile.close()
    
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    print(names)
    print( 'Images #', len(names))
    perm = np.argsort(names)
    count = 0
    for k in perm:
        im = imdata[ids[k]]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        pose = np.linalg.inv(m)

        if count in test_id:
            pose_savefile = os.path.join(save_dir, 'pose', '1_test_%04d.txt'%(count))
        else:
            pose_savefile = os.path.join(save_dir, 'pose', '0_train_%04d.txt'%(count))
        np.savetxt(pose_savefile, pose)
        count = count + 1

if __name__ == "__main__":
    img_dir = sys.argv[1]
    save_dir = sys.argv[2]
    gen_data_from_colmap(img_dir, save_dir)
