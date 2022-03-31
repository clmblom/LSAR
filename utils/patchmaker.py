from PIL import ImageChops, Image
import numpy as np
import torch


def patchify(image, patch_size, overlap=(0, 0)):
    patches = []
    width, height = image.size
    assert patch_size[0] > overlap[0] and patch_size[1] > overlap[1], "Patch needs to be larger than overlap"
    jump_size = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])
    wcs = list(range(0, width - patch_size[0] + 1, jump_size[0]))
    hcs = list(range(0, height - patch_size[1] + 1, jump_size[1]))
    if width % (patch_size[0] - overlap[0]) != 0:
        wcs.append(width - patch_size[0])
    if height % (patch_size[1] - overlap[1]) != 0:
        hcs.append(height - patch_size[1])
    for hc in hcs:
        for wc in wcs:
            patches.append(Patch(image.crop((wc, hc, wc + patch_size[0], hc + patch_size[1])),
                                      (wc, hc),
                                      (patch_size[0], patch_size[1])))
    return patches


def patchify_tensor(image, patch_size, overlap=(0, 0)):
    # image has shape [C, H, W]
    # patch_size is (height, width)
    _, height, width = image.shape
    patch_size = [int(x) for x in patch_size]
    overlap = [int(x) for x in overlap]
    assert patch_size[0] > overlap[0] and patch_size[1] > overlap[1], "Patch needs to be larger than overlap"
    jump_size = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])
    hcs = list(range(0, height - patch_size[0] + 1, jump_size[0]))
    wcs = list(range(0, width - patch_size[1] + 1, jump_size[1]))
    if height % (patch_size[0] - overlap[0]) != 0:
        hcs.append(height - patch_size[0])
    if width % (patch_size[1] - overlap[1]) != 0:
        wcs.append(width - patch_size[1])

    patches = torch.zeros((len(wcs)*len(hcs), 3, patch_size[0], patch_size[1]))
    counter = 0
    for hc in hcs:
        for wc in wcs:
            patches[counter] = image[:, hc:hc+patch_size[0], wc:wc+patch_size[1]]
            counter += 1
    return patches, hcs, wcs


def stitch(patches):
    # Takes a list of Patch
    crop_width, crop_height = patches[-1].crop_size
    width, height = patches[-1].coords
    width += crop_width
    height += crop_height
    image = Image.new('RGB', (width, height))
    for p in patches:
        image.paste(p.patch, (p.coords[0], p.coords[1]))
    return image


def stitch_tensor(patches, hcs, wcs):
    # patches has shape (p, c, h ,w)
    patch_height, patch_width = patches.shape[2], patches.shape[3]
    height = hcs[-1] + patch_height
    width = wcs[-1] + patch_width

    t = torch.zeros((patches.shape[1], height, width))

    counter = 0
    for hc in hcs:
        for wc in wcs:
            t[:, hc:hc+patch_height, wc:wc+patch_width] = patches[counter]
            counter += 1
    return t


def stitch_mask(patches, vote_thresh=0):
    crop_width, crop_height = patches[-1].crop_size
    width, height = patches[-1].coords
    width += crop_width
    height += crop_height
    voting = np.zeros((height, width))
    for p in patches:
        image_tmp = np.array(p.patch).astype(bool)
        voting[p.coords[1]:p.coords[1] + p.crop_size[1], p.coords[0]:p.coords[0] + p.crop_size[0]] += (
                    2 * image_tmp - 1)
    mask = voting >= vote_thresh
    mask = Image.fromarray(mask)
    return mask


def stitch_mask_tensor(patches, hcs, wcs, vote_thresh=0):
    # patches has shape (patch, boolean, height ,width)
    patch_height, patch_width = patches.shape[2], patches.shape[3]
    height = hcs[-1] + patch_height
    width = wcs[-1] + patch_width

    voting = torch.zeros((patches.shape[1], height, width))

    counter = 0
    for hc in hcs:
        for wc in wcs:
            voting[:, hc:hc+patch_height, wc:wc+patch_width] += 2*patches[counter]-1
            counter += 1
    return voting >= vote_thresh


class Patch:
    def __init__(self, patch, coords, crop_size):
        self.patch = patch
        self.coords = coords  # (width, height)
        self.crop_size = crop_size  # (width, height)
