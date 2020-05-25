import numpy as np

def get_edges2face(faces):
    from itertools import combinations
    from sets import Set
    from collections import OrderedDict
    # Returns a structure that contains the faces corresponding to every edge
    edges = OrderedDict()
    for iface, f in enumerate(faces):
        sorted_face_edges = tuple(combinations(sorted(f), 2))
        for sorted_face_edge in sorted_face_edges:
            if edges.has_key(sorted_face_edge):
                edges[sorted_face_edge].faces.add(iface)
            else:
                edges[sorted_face_edge] = lambda:0
                edges[sorted_face_edge].faces = Set([iface])
    return edges

def get_boundary_verts(verts, faces, connected_boundaries=True, connected_faces=False):
    """
     Given a mesh returns boundary vertices
     if connected_boundaries is True it returs a list of lists
     OUTPUT:
        boundary_verts: list of verts
        cnct_bound_verts: list of list containing the N ordered rings of the mesh
    """
    MIN_NUM_VERTS_RING = 10
    # Ordred dictionary
    edge_dict = get_edges2face(faces)
    boundary_verts = []
    boundary_edges = []
    boundary_faces = []
    for edge, (key, val) in enumerate(edge_dict.iteritems()):
        if len(val.faces) == 1:
            boundary_verts += list(key)
            boundary_edges.append(edge)
            for face_id in val.faces:
                boundary_faces.append(face_id)
    boundary_verts = list(set(boundary_verts))
    if not connected_boundaries:
        return boundary_verts
    n_removed_verts = 0
    if connected_boundaries:
        edge_mat = np.array(edge_dict.keys())
        # Edges on the boundary
        edge_mat = edge_mat[np.array(boundary_edges)]

        # check that every vertex is shared by only two edges
        for v in boundary_verts:
            if np.sum(edge_mat == v) != 2:
                import ipdb; ipdb.set_trace();
                raise ValueError('The boundary edges are not closed loops!')

        cnct_bound_verts = []
        while len(edge_mat > 0):
            # boundary verts, indices of conected boundary verts in order
            bverts = []
            orig_vert = edge_mat[0, 0]
            bverts.append(orig_vert)
            vert = edge_mat[0, 1]
            edge = 0
            while orig_vert != vert:
                bverts.append(vert)
                # remove edge from queue
                edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
                edge_mask[edge] = False
                edge_mat = edge_mat[edge_mask]
                edge = np.where(np.sum(edge_mat == vert, axis=1) > 0)[0]
                tmp = edge_mat[edge]
                vert = tmp[tmp != vert][0]
            # remove the last edge
            edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
            edge_mask[edge] = False
            edge_mat = edge_mat[edge_mask]
            if len(bverts) > MIN_NUM_VERTS_RING:
                # add ring to the list
                cnct_bound_verts.append(bverts)
            else:
                n_removed_verts += len(bverts)
    count = 0
    for ring in cnct_bound_verts: count += len(ring)
    assert(len(boundary_verts) - n_removed_verts == count), "Error computing boundary rings !!"

    if connected_faces:
        return (boundary_verts, boundary_faces, cnct_bound_verts)
    else:
        return (boundary_verts, cnct_bound_verts)
