#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from renderer import OrthoTexturedRenderer, OrthoColoredRenderer
from camera import OrthoProjectPoints

GL_NEAREST = 0x2600


class Isomapper():
    def __init__(self, vt, ft, tex_res, bgcolor=np.zeros(3)):
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        self.tex_res = tex_res
        self.f = ft
        self.rn_tex = OrthoTexturedRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=bgcolor)
        self.rn_vis = OrthoColoredRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=np.zeros(3),
                                           num_channels=1)
        self.bgcolor = bgcolor
        self.iso_mask = np.array(self.rn_vis.r)
        print('inside iso')

    def render(self, frame, proj_v, f, visible_faces=None, inpaint=True, inpaint_segments=None):
        h, w, _ = np.atleast_3d(frame).shape
        v2d = proj_v.r
        v2d_as_vt = np.dstack((v2d[:, 0] / w, 1 - v2d[:, 1] / h))[0]

        self.rn_tex.set(texture_image=frame, vt=v2d_as_vt, ft=f)
        tex = np.array(self.rn_tex.r)

        if visible_faces is not None:
            self.rn_vis.set(f=self.f[visible_faces])
            if inpaint:
                visible = cv2.erode(self.rn_vis.r, np.ones((self.tex_res / 100, self.tex_res / 100)))

                if inpaint_segments is None:
                    tex = np.atleast_3d(self.iso_mask) * cv2.inpaint(np.uint8(tex * 255), np.uint8((1 - visible) * 255),
                                                                     3, cv2.INPAINT_TELEA) / 255.
                else:
                    tmp = np.zeros_like(tex)
                    for i in range(np.max(inpaint_segments) + 1):
                        seen = np.logical_and(visible, inpaint_segments == i)
                        part = cv2.inpaint(np.uint8(tex * 255), np.uint8((1 - seen) * 255), 3, cv2.INPAINT_TELEA) / 255.
                        tmp[inpaint_segments == i] = part[inpaint_segments == i]

                    tex = np.atleast_3d(self.iso_mask) * tmp
            else:
                mask = np.atleast_3d(self.rn_vis.r)
                tex = mask * tex + (1 - mask) * self.bgcolor

        return tex