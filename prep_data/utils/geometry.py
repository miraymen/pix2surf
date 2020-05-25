#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import sparse as sp


def laplacian(v, f):
    n = len(v)

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * (ab * ca).sum(axis=1) / np.sqrt(np.sum(np.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * (bc * ab).sum(axis=1) / np.sqrt(np.sum(np.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * (ca * bc).sum(axis=1) / np.sqrt(np.sum(np.cross(ca, bc) ** 2, axis=-1))

    I = np.concatenate((v_a, v_c, v_a, v_b, v_b, v_c))
    J = np.concatenate((v_c, v_a, v_b, v_a, v_c, v_b))
    W = 0.5 * np.concatenate((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a))

    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.spdiags(L * np.ones(n), 0, n, n)

    return L


def get_hres(v, f):
    """
    Get an upsampled version of the mesh.
    OUTPUT:
        - nv: new vertices
        - nf: faces of the upsampled
        - mapping: mapping from low res to high res
    """
    from opendr.topology import loop_subdivider
    (mapping, nf) = loop_subdivider(v, f)
    nv = mapping.dot(v.ravel()).reshape(-1, 3)
    return (nv, nf, mapping)


def barycentric_coordinates(p, q, u, v):
    """
    Calculate barycentric coordinates of the given point
    :param p: a given point
    :param q: triangle vertex
    :param u: triangle vertex
    :param v: triangle vertex
    :return: 1X3 ndarray with the barycentric coordinates of p
    """
    v0 = u - q
    v1 = v - q
    v2 = p - q
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    y = (d11 * d20 - d01 * d21) / denom
    z = (d00 * d21 - d01 * d20) / denom
    x = 1.0 - z - y
    return np.array([x, y, z])
