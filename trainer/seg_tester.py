import torch
import os
from models.metrics import SegmentationMetric
import utils.polyfuncs
from .seg_patch_predict import patch_predict


class Tester:
    def __init__(self, model, config, test_data_loader, device='cpu'):
        self.model = model

        self.config = config

        self.test_loader = test_data_loader
        self.dataloaders = {'test': self.test_loader}
        self.device = device

        self.proj_id = config['name']
        self.save_dir = str(config.save_dir)

        self.test_batch_size = config['test_dataloader']['batch_size']
        if config.resume:
            self._load_checkpoint()
        else:
            assert config.resume, "No model"

    def test_model(self):
        self.test(1, 1)

    def test(self, start_epoch, end_epoch):
        print("Start testing...")
        seg_metric = SegmentationMetric()
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_metric = {'test': dict()}
            for phase in ['test']:
                self.model.eval()
                for i, (images, points, patch_size, patch_overlap, sub_batch) in enumerate(self.dataloaders[phase]):
                    print(f"Image: {i+1}/{len(self.dataloaders[phase])}")
                    mask = patch_predict(self.model, images, patch_size, patch_overlap, sub_batch, self.device)
                    mask_points = utils.polyfuncs.mask_to_contours(mask)
                    seg_metric.update(points, mask_points)
                val_metric_dict = seg_metric.calc_metric()
                epoch_metric[phase] = val_metric_dict
                seg_metric.reset()
            print(epoch_metric)

    def _load_checkpoint(self):
        file_name = os.path.join(self.save_dir, str(self.config.resume))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['state_dict'])

