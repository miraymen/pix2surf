import numpy as np
import cv2
import os
import cPickle as pkl
import argparse

from opendr.serialization import load_mesh
from opendr.camera import ProjectPoints

from utils.iso import Isomapper

from psbody.mesh import Mesh

def main(vt_ft_path, gar_type, mesh_location,  texture_location,
         inpaint_mask, side, cam_file, save_tex_location):

    mesh = Mesh(filename = mesh_location)

    img = cv2.imread(texture_location) / 255.
    img = cv2.resize(img, (1000,1000))

    cam_data = pkl.load(open(cam_file, 'r'))
    cam_y, cam_z = cam_data[gar_type]['cam_y'], cam_data[gar_type]['cam_z']

    camera = ProjectPoints(v=mesh.v, f=np.array([1000, 1000]), c=np.array([1000, 1000]) / 2.,
                           t=np.array([0, cam_y, cam_z]), rt=np.zeros(3), k=np.zeros(5))

    data = pkl.load(open(vt_ft_path, 'r'))[gar_type]
    iso = Isomapper(data['vt'], data['ft'], 2000)

    tex = iso.render(img, camera, mesh.f)
    if side == "front":
        tex_save = tex[500:1500, 0:1000]
    else:
        tex_save = tex[500:1500, 1000:2000]

    inpaint_mask = cv2.imread(inpaint_mask, cv2.IMREAD_UNCHANGED)
    if inpaint_mask is not None:
        tex_save = cv2.inpaint(np.uint8(tex_save * 255), inpaint_mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(save_tex_location, tex_save)

def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--gar_type", type = str, help = "shorts|pants|shirts")
    args.add_argument("--mesh_location", type = str, help = "Location of the registered mesh")
    args.add_argument("--texture_location", type = str, help = "Location of the textured image")
    args.add_argument("--side", type = str, help = "front|back. Whether the texture is to be stored for the front or back")
    args.add_argument("--save_tex_location", type = str, help = "Location of the final texture")
    opt = args.parse_args()

    from os.path import dirname as up
    TWO_UP = up(up(os.path.abspath(__file__)))
    opt.vt_ft_path = os.path.join(TWO_UP, 'assets/vt_ft_file.pkl')
    opt.cam_file = os.path.join(TWO_UP, 'assets/cam_file.pkl')
    opt.inpaint_mask = os.path.join(TWO_UP, 'assets/inpaint_masks/{}_{}.jpg'.format(opt.gar_type, opt.side))

    main(opt.vt_ft_path, opt.gar_type, opt.mesh_location, opt.texture_location,
         opt.inpaint_mask, opt.side, opt.cam_file, opt.save_tex_location)

# python texture_extract.py  --gar_type 'shorts' --mesh_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/meshes/0BL22F004-B12@8.obj' --inpaint_mask '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/assets/inpaint_masks/shorts_front.jpg' --texture_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/gars/0BL22F004-B12@8.jpg' --front True --cam_file '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/assets/cam_file.pkl' --save_tex_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/uvs/0BL22F004-B12@8.jpg'
# python texture_extract.py  --gar_type 'pants' --mesh_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/meshes/sample1.obj' --texture_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/gars/sample1.jpg' --side "front" --save_tex_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/uvs/sample1.jpg'

# python texture_extract.py  --gar_type 'pants' --mesh_location "/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/meshes/sample1.obj" --texture_location "/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/gars/sample1.jpg" --side "front" --save_tex_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/uvs/sample1.jpg'

# python texture_extract.py  --gar_type 'shorts' --mesh_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/meshes/sample1.obj' --texture_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/gars/sample1.jpg' --side "front" --save_tex_location '/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/shorts/front/uvs/0BL22F004-B12@8_ndsf.jpg'
