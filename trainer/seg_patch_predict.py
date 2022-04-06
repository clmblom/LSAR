import utils.patchmaker as pm
import torch


def patch_predict(model, images, patch_size, patch_overlap, sub_batch, device):
    with torch.no_grad():
        patches, wcs, hcs = pm.patchify_tensor(images.squeeze(0),
                                               patch_size=patch_size,
                                               overlap=patch_overlap)
        output = torch.zeros((patches.shape[0], 1, patches.shape[2], patches.shape[3]))
        for sb in range(0, patches.shape[0], sub_batch):
            patch_sub_batch = patches[sb:sb + sub_batch].to(device)
            out = model(patch_sub_batch)
            output[sb:sb + sub_batch] = out.cpu()
        patch_sub_batch = patches[sb:].to(device)
        out = model(patch_sub_batch)
        output[sb:] = out.cpu()
        output = output > 0.5
        mask = pm.stitch_mask_tensor(output, wcs, hcs)
    return mask
