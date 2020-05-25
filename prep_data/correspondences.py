import numpy as np
import argparse
import cv2
import cPickle as pkl
import os


def uv_to_xyz_and_normals(alignment, face_indices_map, b_coords_map):
    pixels_to_set = np.array(np.where(face_indices_map != -1)).T
    x_to_set = pixels_to_set[:, 0]
    y_to_set = pixels_to_set[:, 1]
    b_coords = b_coords_map[x_to_set, y_to_set, :]
    f_coords = face_indices_map[x_to_set, y_to_set].astype(np.int32)
    v_ids = alignment.f[f_coords]

    points = np.tile(b_coords[:, 0], (3, 1)).T * alignment.v[v_ids[:, 0]] + \
             np.tile(b_coords[:, 1], (3, 1)).T * alignment.v[v_ids[:, 1]] + \
             np.tile(b_coords[:, 2], (3, 1)).T * alignment.v[v_ids[:, 2]]
    return points


def main(mesh, sv_file, side, size, fmap_location, bmap_location, cam_file, gar_type):
    from opendr.camera import ProjectPoints
    from psbody.mesh import Mesh

    print(fmap_location)
    cam_data = pkl.load(open(cam_file, 'r'))
    cam_z, cam_y = cam_data[gar_type]['cam_z'], cam_data[gar_type]['cam_y']

    mesh = Mesh(filename = mesh)
    fmap = np.load(fmap_location)
    bmap = np.load(bmap_location)

    cam = ProjectPoints(v=mesh.v, t=np.array([0, cam_y, cam_z]), rt=np.zeros(3), f=[1000, 1000],
                        c=[1000 / 2., 1000 / 2.], k=np.zeros(5))

    points = uv_to_xyz_and_normals(mesh, fmap, bmap)
    cam.v = points
    projection = cam.r.astype(np.int32)
    projection[projection > 999] = 999

    projection = np.fliplr(np.around(projection.squeeze())).astype(np.int32)

    pixels_to_set = np.array(np.where(fmap != -1)).T
    x_to_set = pixels_to_set[:, 0]
    y_to_set = pixels_to_set[:, 1]

    cords_ret = -999 * np.ones((fmap.shape[0], fmap.shape[1], 2))
    cords_ret[x_to_set, y_to_set, : ] = projection

    cords_ret = cords_ret.astype('float64')

    cords_ret = 2*((cords_ret) / 999) - 1

    cords_ret = np.flip(cords_ret, 2)

    if side == 'front':
        cords_ret = cords_ret[500:1500, 0 : 1000]
    else:
        cords_ret = cords_ret[500:1500, 1000: 2000]

    cords_ret = cv2.resize(cords_ret, (size, size))
    np.save(sv_file, cords_ret)

def str2bool(v):
    return v.lower() in ('true')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--gar_type", type = str, help = "Garment type")
    args.add_argument("--mesh_file", type = str, help = "the location of the mesh")
    args.add_argument("--sv_cords", type = str, help = "Location where coordinates are to be stored")
    args.add_argument("--side", type = str, help = "Whether it is the front or the back we are dealing with front | back")
    args.add_argument("--size", type = int, default = 256, help = "The size of the final image")

    args = args.parse_args()

    from os.path import dirname as up
    TWO_UP = up(up(os.path.abspath(__file__)))
    args.fmap_location = os.path.join(TWO_UP, 'assets/fmaps/{}.npy'.format(args.gar_type))
    args.bmap_location = os.path.join(TWO_UP, 'assets/bmaps/{}.npy'.format(args.gar_type))
    args.cam_file = os.path.join(TWO_UP, 'assets/cam_file.pkl')


    main(args.mesh_file, args.sv_cords, args.side, args.size, args.fmap_location, args.bmap_location, args.cam_file, args.gar_type)

# python /BS/composition_gan2/work/thesis/PROJECTS/correspondences/main.py /BS/composition_gan2/work/thesis/DATA/Mesh_saved/front-all/1006902_14911_7.obj /BS/composition_gan2/work/thesis/DATA/T-downs/All-front/1006902_14911_7.jpg /BS/composition_gan2/work/thesis/DATA/coords/test.npy
# python correspondences.py  --gar_type 'shorts' --mesh_file '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/meshes/0BL22F004-B12@8.obj' --sv_cords '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/cords/0BL22F004-B12@8.npy' --side "front"