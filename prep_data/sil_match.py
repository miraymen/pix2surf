#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import cPickle as pkl
import numpy as np
from PIL import Image
import json
import argparse
import os
from os.path import dirname as up

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.filters import gaussian_pyramid
import chumpy as ch
from psbody.mesh import Mesh

from utils.smpl_paths import SmplPaths
from utils.ch_smpl import Smpl
from utils import im


glob_var = 0

def get_submesh(verts, faces, verts_retained,  min_vert_in_face=2):
    '''
        Given a mesh, create a (smaller) submesh
        indicate faces or verts to retain as indices or boolean
        @return new_verts: the new array of 3D vertices
                new_faces: the new array of faces
                bool_faces: the faces indices wrt the input mesh
                vetex_ids: the vertex_ids of the new mesh wrt the input mesh
        '''

    # Transform indices into bool array
    if verts_retained.dtype != 'bool':
        vert_mask = np.zeros(len(verts), dtype=bool)
        vert_mask[verts_retained] = True
    else:
        vert_mask = verts_retained

    # Faces with at least min_vert_in_face vertices
    bool_faces = np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) > min_vert_in_face


    new_faces = faces[bool_faces]
    # just in case additional vertices are added
    vertex_ids = list(set(new_faces.ravel()))

    oldtonew = -1 * np.ones([len(verts)])
    oldtonew[vertex_ids] = range(0, len(vertex_ids))

    new_verts = verts[vertex_ids]
    new_faces = oldtonew[new_faces].astype('int32')

    return (new_verts, new_faces, bool_faces, vertex_ids)


def uv_to_v_ids(alignment, face_indices_map):
    """
    Finds all the vertex ids in the face map
    alignment: mesh with .v and .f attributes
    face_indices_map: fmap we are dealing with
    return v_ids: vertex ids
    """
    pixels_to_set = np.array(np.where(face_indices_map != -1)).T
    x_to_set = pixels_to_set[:, 0]
    y_to_set = pixels_to_set[:, 1]
    f_coords = face_indices_map[x_to_set, y_to_set].astype(np.int32)
    v_ids = alignment.f[f_coords]

    return v_ids


def get_part(front, path_fmap, path_mesh):
    """
    Gets the submesh corresponding to the part of a mesh
    front: True if the part corresponds to front
    path_fmap: location of fmap
    path_mesh: location of template mesh
    """
    fmap = np.load(path_fmap)

    if front:
        fmap_cut = fmap[:, 0: 1000]
    else:
        fmap_cut = fmap[:, 1000:2000]

    mesh = Mesh(filename=path_mesh)

    verts = uv_to_v_ids(mesh, fmap_cut)
    verts = np.array(list(set(verts.ravel())))

    submesh_verts, submesh_faces, _, v_ids = get_submesh(mesh.v, mesh.f, verts_retained=verts)


    return submesh_verts, submesh_faces, v_ids



def laplacian(part_mesh):
    """ Compute laplacian operator on part_mesh. This can be cached.
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    from psbody.mesh.topology.connectivity import get_vert_connectivity
    import numpy as np
    connectivity = get_vert_connectivity(part_mesh)
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = sp.eye(connectivity.shape[0]) - lap
    return lap



def get_callback(rend, mask, smpl, v_ids_template, v_ids_sides, faces_template, faces_side, display, disp_mesh_side, disp_mesh_whl, mv, mv2, save_dir, show):
    """
    :param rend: debug renderer
    :param mask: mask of the garment tshirt
    :param smpl: SMPL cutout we are deforming
    :param v_ids_template: vertex ids of the whole template mesh as cut-out from SMPL
    :param v_ids_sides: vertex ids of the front or back of the template mesh
    :param faces_template: the faces of the cut out template from SMPL
    :param faces_side: the faces of the side cut out from the template
    :param display: Whether there should be any debug checks
    :param disp_mesh: display the side cutout of the mesh when its deforming
    :param disp_mesh_whl: display the whole cutout mesh when its deforming
    :param mv: renderer for part
    :param mv2: renderer for whole
    :param save_dir: Directory where optimization overlaps are to be saved
    :param show: Whether debug overlaps are to be displayed
    :return: nothing. just for visualization
    """
    from psbody.mesh import MeshViewer
    if display:
        rend.set(v=smpl[v_ids_template][v_ids_sides], background_image=mask)
        rend.vc.set(v=smpl[v_ids_template][v_ids_sides])

        def cb(_):

            debug = np.array(rend.r)
            import time
            global glob_var

            if save_dir != None:
                cv2.imwrite(save_dir + '/' + str(glob_var) + '.jpg', 255 * (1 - debug))

            glob_var = glob_var + 1
            if show:
                im.show(debug, id='Init', waittime=1)

            if disp_mesh_side:
                m = Mesh(v=smpl[v_ids_template][v_ids_sides],f=faces_side)
                mv.set_dynamic_meshes([m])

            if disp_mesh_whl:
                m2 = Mesh(v = smpl[v_ids_template], f = faces_template)
                mv2.set_dynamic_meshes([m2])
        return cb
    else:

        return None


def get_callback_ref(rend, mask, vertices, display,  v_ids_sides, faces_template, faces_side,  disp_mesh_side, disp_mesh_whl, mv, mv2, save_dir, show):
    """
    :param rend: debug renderer
    :param mask: mask of the garment tshirt
    :param vertices: SMPL cutout we are deforming
    :param display: Whether there should be any debug checks
    :param v_ids_sides: vertex ids of the front or back of the template mesh
    :param faces_template: the faces of the cut out template from SMPL
    :param faces_side: the faces of the side cut out from the template
    :param disp_mesh: display the side cutout of the mesh when its deforming
    :param disp_mesh_whl: display the whole cutout mesh when its deforming
    :param mv: renderer for part
    :param mv2: renderer for whole
    :param save_dir: Directory where optimization overlaps are to be saved
    :param show: Whether debug overlaps are to be displayed
    :return: nothing. just for visualization
    """
    if display:
        rend.set(v=vertices[v_ids_sides], background_image=mask)
        rend.vc.set(v=vertices[v_ids_sides])

        def cb(_):

            debug = np.array(rend.r)
            global glob_var

            if save_dir != None:
                cv2.imwrite(save_dir + '/' + str(glob_var) + '.jpg', 255 * (1 - debug))

            glob_var = glob_var + 1

            if show:
                im.show(debug, id='Refinement', waittime=1)

            if disp_mesh_side:
                m = Mesh(v=vertices[v_ids_sides], f=faces_side)
                mv.set_dynamic_meshes([m])

            if disp_mesh_whl:
                m2 = Mesh(v = vertices, f = faces_template)
                mv2.set_dynamic_meshes([m2])

        return cb
    else:

        return None


def get_pose_prior(init_pose_path, gar_type):
    tgt_apose = pkl.load(open(init_pose_path, 'rb'))
    tgt_apose = tgt_apose['pose']
    if gar_type == 'shorts':
        tgt_apose[5] = 0.3
        tgt_apose[8] = -0.3

    return tgt_apose


def compute_boundaries(verts, faces):
    """
    Compute boundary rings.
    :param verts: vertex locations
    :param faces: faces of mesh
    :return all the rings in the garment
    """
    from utils.boundary import get_boundary_verts
    [dum, garm_rings] = get_boundary_verts(verts, faces)
    return garm_rings


def smooth_rings(gar_rings, verts):
    """
    :param gar_rings:
    :param verts:
    :returns
    """
    error = ch.zeros([0, 3])

    for ring in gar_rings:
        N = len(ring)
        aring = np.array(ring)
        ring_0 = aring[np.arange(0, N)]
        ring_m1 = aring[np.array([N - 1] + range(0, N - 1))]
        ring_p1 = aring[np.array(range(1, N) + [0])]
        err_ring = verts[ring_m1] - 2 * verts[ring_0] + verts[ring_p1]
        error = ch.vstack([error, err_ring])

    error = ch.vstack([error, err_ring])
    return error


def get_verts_rings(gar_rings, verts, v_ids_side):

    lengths = [len(ring) for ring in gar_rings]
    max_len_index = lengths.index(max(lengths))

    gar_ring_largest = gar_rings[max_len_index]
    intersection = list(set(gar_ring_largest) & set(v_ids_side))
    position_largest_ring = verts[intersection]

    return position_largest_ring

def get_rings_error(cam, max_y_val):
    return cam[:, 1] - max_y_val * np.ones(cam[:, 1].shape)


def get_dist_tsfs(mask):
    """
    :param mask: binary mask
    :return: dist_i, dist_o, dif_mask
    """
    dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2

    dist_i = cv2.distanceTransform(np.uint8(mask * 255), dst_type, 5) - 1
    dist_i[dist_i < 0] = 0
    dist_i[dist_i > 50] = 50
    dist_o = cv2.distanceTransform(255 - np.uint8(mask * 255), dst_type, 5) #zero where the binary image is 1
    dif_mask = mask.copy()
    dif_mask[mask == 1] = 0.5

    return dist_i, dist_o, dif_mask

def compute_iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)

    iou_score = float(np.sum(intersection)) / float(np.sum(union))

    return iou_score

def init_fit(opt, dist_o, dist_i, dif_mask, rn_m, smpl_h,
             v_ids_template, faces_template, debug_rn, v_ids_side, faces_side,
             joints_list
             ):

    range_joint = []

    for item in joints_list:
        range_joint.append(3 * int(item))
        range_joint.append(3 * int(item) + 1)
        range_joint.append(3 * int(item) + 2)

    tgt_pose = get_pose_prior(init_pose_path=opt.init_pose_path, gar_type = opt.gar_type)

    # ============================================
    #                 FIRST STAGE
    # ============================================
    from psbody.mesh import MeshViewer

    if opt.disp_mesh_side:
        mv = MeshViewer()
    else:
        mv = None
    if opt.disp_mesh_whl:
        mv2 = MeshViewer()
    else:
        mv2 = None

    if opt.init_first_stage == "Trans":
        x0 = [smpl_h.trans]

    elif opt.init_first_stage == 'Pose':
        x0 = [smpl_h.trans, smpl_h.pose[range_joint]]
        # x0 = [smpl_h.trans]

    elif opt.init_first_stage == 'Shape':
        x0 = [smpl_h.trans, smpl_h.betas]

    elif opt.init_first_stage == 'Both':
        x0 = [smpl_h.trans, smpl_h.betas, smpl_h.pose[range_joint]]

    E = {
        'mask': gaussian_pyramid(rn_m * dist_o * opt.init_fst_wt_dist_o + (1 - rn_m) * dist_i, n_levels=4, normalization='size') * opt.init_fst_wt_mask
    }

    if opt.init_fst_wt_betas:
        E['beta_prior'] = ch.linalg.norm(smpl_h.betas) * opt.init_fst_wt_betas

    if opt.init_fst_wt_pose:
        E['pose'] = (smpl_h.pose - tgt_pose) * opt.init_fst_wt_pose

    ch.minimize(
        E,
        x0,
        method='dogleg',
        options={
            'e_3': .0001,
            'disp': True
        },
        callback=
        get_callback(
            rend = debug_rn, mask = dif_mask, smpl=smpl_h, v_ids_template=v_ids_template, v_ids_sides = v_ids_side,
            faces_template = faces_template, faces_side = faces_side, display = opt.display, disp_mesh_side = opt.disp_mesh_side,
            disp_mesh_whl=opt.disp_mesh_whl, mv = mv, mv2 = mv2, save_dir = opt.save_opt_images, show = opt.show)

        )


    # ===============================================
    #                 SECOND STAGE
    # ===============================================

    if opt.init_second_stage != "None":
        if opt.init_second_stage == 'Pose':
            x0 = [smpl_h.trans, smpl_h.pose[range_joint]]

        elif opt.init_second_stage == 'Shape':
            x0 = [smpl_h.trans, smpl_h.betas]

        elif opt.init_second_stage == 'Both':
            x0 = [smpl_h.trans, smpl_h.betas, smpl_h.pose[range_joint]]

        E = {
            'mask': gaussian_pyramid(rn_m * dist_o * opt.init_sec_wt_dist_o + (1 - rn_m) * dist_i, n_levels=4,
                                     normalization='size') * opt.init_sec_wt_mask
        }

        if opt.init_sec_wt_betas:
            E['beta_prior'] = ch.linalg.norm(smpl_h.betas) * opt.init_sec_wt_betas

        if opt.init_sec_wt_pose:
            E['pose'] = (smpl_h.pose - tgt_pose) * opt.init_sec_wt_pose

        ch.minimize(
            E,
            x0,
            method='dogleg',
            options={
                'e_3': .0001
            },
            callback = get_callback
            (
                rend = debug_rn, mask = dif_mask, smpl=smpl_h, v_ids_template=v_ids_template, v_ids_sides = v_ids_side,
                faces_template = faces_template, faces_side = faces_side, display = opt.display, disp_mesh_side = opt.disp_mesh_side,
                disp_mesh_whl=opt.disp_mesh_whl, mv = mv, mv2 = mv2, save_dir = opt.save_opt_images, show = opt.show
            )
        )


    temp_params = {'pose': smpl_h.pose.r, 'betas': smpl_h.betas.r, 'trans': smpl_h.trans.r,
                   'v_personal': smpl_h.v_personal.r}

    part_mesh = Mesh(smpl_h.r[v_ids_template], faces_template)

    return part_mesh, temp_params


def final_fit(
        opt ,
        part_mesh, v, v_offset, dist_o,dist_i, smpl_h_ref , rn_m ,
        debug_rn, dif_mask , v_ids_template,faces_template ,
        v_ids_side , faces_side , max_y , proj_cam ,
        ref_joint_list_coup,
):
    if opt.disp_mesh_side:
        mv = MeshViewer()
    else:
        mv = None
    if opt.disp_mesh_whl:
        mv2 = MeshViewer()
    else:
        mv2 = None

    import scipy.sparse as sp
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=500)[0]

    tgt_pose = get_pose_prior(init_pose_path=opt.init_pose_path, gar_type = opt.gar_type)

    E = {
        'mask': gaussian_pyramid(rn_m * dist_o * opt.ref_wt_dist_o + (1 - rn_m) * dist_i , n_levels=4,
                                 normalization='size') * opt.ref_wt_mask}

    x0 = [v_offset]
    if opt.ref_wt_coup:
        # x0 = [smpl_h_ref.trans, smpl_h_ref.betas, v_offset]
        E['coupling'] = (v + v_offset - smpl_h_ref[v_ids_template]) * opt.ref_wt_coup

        if opt.ref_wt_shp:
            E['beta_prior'] = ch.linalg.norm(smpl_h_ref.betas) * opt.ref_wt_shp

        if opt.ref_wt_pose:

            E['pose'] = (smpl_h_ref.pose - tgt_pose) * opt.ref_wt_pose

        if ref_joint_list_coup != None:
            range_joint = []
            for item in ref_joint_list_coup:
                range_joint.append(3 * int(item))
                range_joint.append(3 * int(item) + 1)
                range_joint.append(3 * int(item) + 2)

            x0 = x0 + [smpl_h_ref.pose[range_joint]]

        if opt.ref_use_betas:
            x0 = x0 + [smpl_h_ref.betas]

    if opt.ref_wt_proj:
        error_bd = get_rings_error(proj_cam, max_y)
        E['proj_bd'] = error_bd * opt.ref_wt_proj

    if opt.ref_wt_bd:
        gar_rings = compute_boundaries(v + v_offset, faces_template)
        error = smooth_rings(gar_rings, v + v_offset)
        E['boundary'] = error * opt.ref_wt_bd

    if opt.ref_wt_lap:
        lap_op = np.asarray(laplacian(part_mesh).todense())
        lap_err = ch.dot(lap_op, v + v_offset)
        E['laplacian'] = lap_err * opt.ref_wt_lap


    ch.minimize(
        E,
        x0,
        method='dogleg',
        options={
            'e_3': .000001, 'sparse_solver': sparse_solver
        },
        callback=get_callback_ref(rend = debug_rn, mask = dif_mask, vertices = v+v_offset, display = opt.display,
                                  v_ids_sides = v_ids_side, faces_template = faces_template, faces_side = faces_side,
                                  disp_mesh_side= opt.disp_mesh_side, disp_mesh_whl = opt.disp_mesh_whl, save_dir = opt.save_opt_images,
                                  mv = mv, mv2 = mv2, show = opt.show)
    )

    final_mask = rn_m.r

    mask = dif_mask.copy()
    mask[dif_mask == 0.5] = 1

    final_iou = compute_iou(mask, final_mask)

    return v+v_offset, final_iou


def get_cam_rend(verts, faces, cam_y, cam_z):

    frustum = {'near': 0.1, 'far': 1000., 'width': 1000, 'height': 1000}

    camera = ProjectPoints(v=verts, t=np.array([0, cam_y, cam_z]), rt=np.zeros(3), f=[1000, 1000],
                           c=[1000 / 2., 1000 / 2.], k=np.zeros(5))

    rn_m = ColoredRenderer(camera=camera, v=verts, f=faces, vc=np.ones_like(verts),
                           frustum=frustum, bgcolor=0, num_channels=1)

    return camera, rn_m

def get_mask(path):
    """
    Reads the mask
    """
    mask = np.array(Image.open(path))

    mask = mask.astype(np.float)
    mask = mask / 255
    mask = np.round(mask)

    return mask

def get_max_min_mask(mask):
    """
    :param mask: binary mask
    :return: min and max y location of the binary mask
    """
    locs = np.array(np.where(mask == 1)).T
    y_locs = locs[:, 0]

    max_y = np.max(y_locs)
    min_y = np.min(y_locs)

    return min_y, max_y

def init_smpl(gender, init_pose_path, gar_file_path, template_file_pkl_path, gar_type):
    """

    """
    dp = SmplPaths(gender=gender)

    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    tgt_apose = pkl.load(open(init_pose_path, 'rb'))

    tgt_apose = tgt_apose['pose']
    if gar_type == 'shorts':
        tgt_apose[5] = 0.3
        tgt_apose[8] = -0.3

    smpl_h.trans[:] = 0

    gar = pkl.load(open(gar_file_path))
    verts = gar[gar_type]['vert_indices']

    data = pkl.load(open(template_file_pkl_path))
    v_personal = np.array(data['v_personal'])
    smpl_h.v_personal[verts] = v_personal

    smpl_h.pose[:] = tgt_apose

    return smpl_h


def main(opt):
    if opt.save_json_file != "None":
        dict_opts = vars(opt)
        with open(opt.save_json_file, 'w') as f:
            json.dump(dict_opts, f, sort_keys=True, indent=4)

    # Initialize joints used
    if not(opt.init_pose_joints == "None"):
        init_joints_list = [int(item) for item in opt.init_pose_joints.split(',')]
    else:
        init_joints_list = None

    if not(opt.ref_joint_list_coup ==  "None"):
        ref_joints_list = [int(item) for item in opt.ref_joint_list_coup.split(',')]
    else:
        ref_joints_list = None

    # GET FILES
    TWO_UP = up(up(os.path.abspath(__file__)))

    opt.init_pose_path = os.path.join(TWO_UP, 'assets/apose.pkl')
    opt.fmap_path = os.path.join(TWO_UP, 'assets/fmaps/{}.npy'.format(opt.gar_type))
    opt.cam_file = os.path.join(TWO_UP, 'assets/cam_file.pkl')
    opt.template_mesh_path = os.path.join(TWO_UP, 'assets/init_meshes/{}.obj'.format(opt.gar_type))
    opt.template_mesh_pkl_path = os.path.join(TWO_UP, 'assets/init_meshes/{}.pkl'.format(opt.gar_type))
    opt.gar_file_path = os.path.join(TWO_UP, 'assets/gar_file.pkl')

    # Get camera params
    cam_data = pkl.load(open(opt.cam_file, 'r'))
    opt.cam_z, opt.cam_y = cam_data[opt.gar_type]['cam_z'], cam_data[opt.gar_type]['cam_y']

    # Get vertex and face ids
    gar = pkl.load(open(opt.gar_file_path))
    v_ids_template = gar[opt.gar_type]['vert_indices']
    faces_template = gar[opt.gar_type]['f']

    # Get vertex ids and faces for the template
    vertices_template, faces_side, v_ids_side = get_part(opt.front, opt.fmap_path, opt.template_mesh_path)

    # Initialize the SMPL template
    template_smpl = init_smpl(gender = opt.gender, init_pose_path = opt.init_pose_path, gar_file_path=opt.gar_file_path, template_file_pkl_path=opt.template_mesh_pkl_path, gar_type = opt.gar_type)

    # Get masks and distance transforms
    mask = get_mask(opt.mask_file)
    dist_i, dist_o, dif_mask = get_dist_tsfs(mask)
    # ==============================================
    #               FIRST STAGE
    # ==============================================

    ## Initialize camera and renderer
    ## Initialize debug camera and renderer

    debug_cam_init, debug_rend_init = get_cam_rend(verts = template_smpl[v_ids_template][v_ids_side], faces = faces_side, cam_y = opt.cam_y, cam_z = opt.cam_z)

    opt_cam_init, opt_rend_init = get_cam_rend(verts =template_smpl [v_ids_template][v_ids_side], faces = faces_side, cam_y = opt.cam_y, cam_z = opt.cam_z)

    part_mesh, temp_params = init_fit(opt = opt,
                                      dist_o = dist_o, dist_i = dist_i, dif_mask = dif_mask, rn_m = opt_rend_init, smpl_h = template_smpl, v_ids_template = v_ids_template,
                                      faces_template = faces_template, debug_rn = debug_rend_init,  v_ids_side = v_ids_side, faces_side = faces_side,
                                      joints_list = init_joints_list
                                      )

    # ==============================================
    #               REFINEMENT STAGE
    # ==============================================

    v = np.array(part_mesh.v)
    v_offset = ch.zeros(v.shape)

    dp = SmplPaths(gender= opt.gender)
    smpl_h_refine =  Smpl(dp.get_hres_smpl_model_data())

    data = temp_params
    smpl_h_refine.pose[:] = data["pose"]
    smpl_h_refine.trans[:] = data["trans"]
    smpl_h_refine.betas[:] = data["betas"]
    smpl_h_refine.v_personal[:] = data["v_personal"]


    ## Initialize second camera and renderer
    ## Initialize second debug camera and renderer

    debug_cam_ref, debug_rend_ref = get_cam_rend(verts = v[v_ids_side] + v_offset[v_ids_side], faces = faces_side, cam_y = opt.cam_y, cam_z = opt.cam_z)
    opt_cam_ref, opt_rend_ref = get_cam_rend(verts = v[v_ids_side] + v_offset[v_ids_side], faces = faces_side, cam_y = opt.cam_y, cam_z = opt.cam_z)

    ## Rings and camera for the projection error
    gar_rings = compute_boundaries(v + v_offset, faces_template)
    position_largest_ring = get_verts_rings(gar_rings = gar_rings, verts = v + v_offset, v_ids_side = v_ids_side )
    proj_cam_ref, _ = get_cam_rend(verts = position_largest_ring, faces = faces_side, cam_y = opt.cam_y, cam_z = opt.cam_z)
    max_y, min_y = get_max_min_mask(mask)

    final_verts, final_iou = final_fit(
        opt = opt, part_mesh = part_mesh, v = v, v_offset = v_offset, dist_o = dist_o, dist_i = dist_i,
        smpl_h_ref = smpl_h_refine, rn_m = opt_rend_ref, debug_rn = debug_rend_ref, dif_mask = dif_mask,
        v_ids_template = v_ids_template,faces_template = faces_template, v_ids_side = v_ids_side,
        faces_side = faces_side, max_y = max_y, proj_cam = proj_cam_ref, ref_joint_list_coup= ref_joints_list
    )

    mesh_sv = Mesh(v=final_verts, f=faces_template)
    mesh_sv.write_obj(opt.save_file)

    if opt.save_iou_file != "None":
        with open(opt.save_iou_file, 'a+') as fp:
            fp.write('{} , {} \n'.format(opt.save_file, str(final_iou)))

        fp.close()

def str2bool(v):
    return v.lower() in ('true')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_from_json", type = str, help = "None| location of json file from which arguments are to be loaded")
    #===================================
    #           FILE LOCATIONS
    #===================================
    parser.add_argument("--save_file", type=str, help="location of final obj file saved")
    parser.add_argument("--mask_file", type=str, help = "location of initial mask file")
    parser.add_argument("--save_opt_images", type = str, default = "None", help = "Determines whether optimization silhouette images are saved or not. If 'None', then nothing gets saved" )
    parser.add_argument("--save_iou_file", type = str, default = "None", help = "location of the txt file where mean iou is to stored")
    parser.add_argument("--save_json_file", type = str, default = "None", help = "location of the json file in which all the input arguments are to be stored")
    #==============================
    #           FLAGS
    #==============================
    parser.add_argument("--front", type=str2bool, default=True, help="Whether to use the front or the back")
    parser.add_argument("--display", type=str2bool, default=True, help="Whether to display callback or not- global var controls whether callback is none or there")
    parser.add_argument("--show", type = str2bool, default = True, help = "Whether or not to display the silhouette matching or not")
    parser.add_argument("--disp_mesh_whl", type = str2bool, default = False, help = "Display the whole mesh in debugging as well or not")
    parser.add_argument("--disp_mesh_side", type = str2bool, default = False, help = "Display the mesh corresponding to the side while debugging or not")
    parser.add_argument("--gender", type = str, default = "neutral" , help = "male|neutral. If chosen male only the first ten coefficients will be used")
    parser.add_argument("--gar_type", type = str, default = "shorts", help = "Garment category we are dealing with shorts|pants|shirts")
    #================================
    #       INIT STAGE FLAGS
    #================================
    parser.add_argument("--init_first_stage" , type = str, default = "Pose", help = "Pose|Shape|Trans|Both")
    parser.add_argument("--init_second_stage", type = str, default = "Both", help = "Pose|Shape|Both|None all as a string")
    parser.add_argument("--init_pose_joints", type = str, default = "1,2", help = "list of joints used for optimization in the first stage")
    parser.add_argument("--init_fst_wt_betas", type = float, default = 4, help = "Initialization first phase weight of shape prior")
    parser.add_argument("--init_fst_wt_pose", type = float, default = 60, help = "Initialization first phase weight of pose prior")

    parser.add_argument("--init_sec_wt_betas", type = float, default = 8, help = "Initialization second phase weight of shape prior")
    parser.add_argument("--init_sec_wt_pose", type = float, default = 40, help = "Initialization second phase weight of pose prior")
    parser.add_argument("--init_fst_wt_dist_o", type = float, default = 2, help = "Initialization first phase, weight of the dist_o term")
    parser.add_argument("--init_fst_wt_mask", type = float, default = 100, help = "Initialization first phase, weight of the mask term")
    parser.add_argument("--init_sec_wt_dist_o", type = float, default = 4, help = "Initialization second phase, weight of the dist_o term")
    parser.add_argument("--init_sec_wt_mask", type = float, default = 100, help = "Initialization second phase, weight of the mask term")
    #===================================
    #       REFINEMENT STAGE FLAGS
    #===================================
    parser.add_argument("--ref_wt_dist_o", type = float, default = 200, help = "Weight of the dist_o term in the refinement")
    parser.add_argument("--ref_wt_mask", type = float, default = 100, help = "weight of the mask term in the refinement")
    parser.add_argument("--ref_wt_coup", default = 0, type=float, help="Weight of coupling term")
    parser.add_argument("--ref_wt_proj", default = 0.1, type=float, help="Weight of the projection term")
    parser.add_argument("--ref_wt_bd", default = 4, type = float, help = "weight of the boundary smoothing term")
    parser.add_argument("--ref_wt_lap", default = 1000, type = float, help = "weight of the laplacian term in the refinement step")
    parser.add_argument("--ref_joint_list_coup", type = str, default = "1,2", help = "optimize pose during the refinement")
    parser.add_argument("--ref_wt_shp", type = float, default = 0, help = "Whether to use shape regularization in the refinement term")
    parser.add_argument("--ref_wt_pose", type = float, default = 0, help = "Whether to use pose prior in the final refinement term")
    parser.add_argument("--ref_use_betas", type = str2bool, default = False, help = "Whether to use betas as variables in the refinement")

    args = parser.parse_args()
    print(args)
    if args.load_from_json:
        with open(args.load_from_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
            print(args)

    main(opt = args)

