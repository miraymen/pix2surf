import torch
import numpy


def flip(tensor, dim):
    tensor1 = tensor.cpu()
    t2 = numpy.flip(tensor1.numpy(), dim).copy()
    t2 = torch.from_numpy(t2)
    return t2

def tensor2image(image):
    image_new = image.detach().clone().cpu()
    image_new = flip(image_new, 0)
    image_new = image_new.permute(1, 2, 0)
    image_new = image_new.numpy()
    image_new = (image_new * 255).astype('uint8')
    return image_new

def sil2image(image):
    image_new = image.detach().clone().cpu()
    image_new = image_new.numpy()
    image_new = (image_new * 255).astype('uint8')
    image_new = numpy.dstack((image_new, image_new, image_new))
    return image_new

def binarizeimage(image):
    binary = torch.zeros(image.size())
    binary[image >= 0.5 ] = 1
    return binary

def str2bool(v):
    return v.lower() in ('true')


def sv_dsk_segs(image, sil, out_fg, out_bcg):

    image = tensor2image(image)
    sil = sil2image(sil)
    out_fg = sil2image(out_fg)
    out_bcg = sil2image(out_bcg)

    out_fg = out_fg / 255
    imag2 = image * out_fg
    out_fg = out_fg * 255

    horz1 = numpy.hstack((image, imag2, sil, out_fg, out_bcg))
    return horz1

def sv_dsk_map(visual_items):
    uv_out = tensor2image(visual_items["uv_out"])
    rend = tensor2image(visual_items["gar"])
    uv = tensor2image(visual_items["uv"])


    horz1 = numpy.hstack((rend, uv_out, uv))
    return horz1
