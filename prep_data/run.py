import argparse
import os
from os.path import dirname as up

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_file", type = str, help = "Location of image file", default = "/BS/composition_gan2/work/thesis_ext/CODE_RELEASE/train/data/pants/front/gars/sample1.jpg")
    ap.add_argument("--gar_type", type = str, help = "shorts|shirts|pants", default = 'pants')
    ap.add_argument("--side", type = str, help = "front|back", default = 'front')
    args = ap.parse_args()

    file = os.path.basename(args.img_file)
    PATH = up(up(os.path.abspath(args.img_file)))

    paths = ['meshes', 'uvs', 'cords', 'masks']
    paths_dir = {}
    paths_file = {}

    for val in paths:
        paths_dir[val+'_dir'] = os.path.join(PATH, val)

        if not os.path.exists(paths_dir[val+'_dir']):
            os.makedirs(paths_dir[val+'_dir'])

        if val == 'cords':
            paths_file[val + '_file'] = os.path.join(paths_dir[val+'_dir'], file).replace('.jpg', '.npy')
        elif val == 'meshes':
            paths_file[val + '_file'] = os.path.join(paths_dir[val+'_dir'], file).replace('.jpg', '.obj')
        else:
            paths_file[val + '_file'] = os.path.join(paths_dir[val+'_dir'], file)

    str_sil = 'python sil_match.py --save_file {} --mask_file {} --load_from_json {}'.format(paths_file['meshes_file'], paths_file['masks_file'], './json_params/'+args.gar_type+'_'+args.side)
    str_cors = 'python correspondences.py --gar_type {} --mesh_file {} --sv_cords {} --side {}'.format(args.gar_type, paths_file['meshes_file'], paths_file['cords_file'], args.side)
    str_txt = 'python texture_extract.py --gar_type {} --mesh_location {} --texture_location {} --side {} --save_tex_location {}'.format(args.gar_type, paths_file['meshes_file'], args.img_file, args.side, paths_file['uvs_file'])

    os.system(str_sil)
    os.system(str_cors)
    os.system(str_txt)
