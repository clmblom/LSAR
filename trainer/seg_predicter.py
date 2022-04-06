import torch
import os
import utils.polyfuncs as pf
from utils.page_writer import PageWriter
from .seg_patch_predict import patch_predict


class Predicter:
    def __init__(self, model, config, inf_data_loader, device='cpu'):
        self.model = model
        self.config = config
        self.dataloaders = {'inf': inf_data_loader}

        self.device = device

        self.save_dir = str(config.save_dir)
        if config.resume:
            self._load_checkpoint()
        else:
            assert config.resume, "Model missing from -r"

    def predict(self):
        print("Starting inference...")
        self.model.eval()
        dataset = self.dataloaders['inf'].dataset
        for i, (images, indx, patch_size, patch_overlap, sub_batch) in enumerate(self.dataloaders['inf']):
            file_name = dataset.data['name'][indx]
            print("Segmenting", file_name)
            mask = patch_predict(self.model, images, patch_size, patch_overlap, sub_batch, self.device)
            mask_points = pf.mask_to_contours(mask)
            mask_polygons = pf.contours_to_polygons(mask_points, make_square=False, rotated=False, shrink_factor=20)
            mask_midlines = pf.polygons_to_midlines(mask_polygons)
            _, height, width = mask.shape

            pw = PageWriter()
            pw.create_base()
            page = pw.add_page(file_name, width, height)
            tr = pw.add_text_region(page, [(0, 0), (width, 0), (width, height), (0, height)])
            mask_points_expanded = pf.polygons_to_points(mask_polygons)
            for poly_coords, midline_coords in zip(mask_points_expanded, mask_midlines):
                tl = pw.add_text_line(tr, poly_coords)
                pw.add_base_line(tl, midline_coords)
            pw.write_xml(os.path.join('output', 'segmentation', file_name + '.xml'))

    def _load_checkpoint(self):
        file_name = os.path.join(self.save_dir, str(self.config.resume))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['state_dict'])
