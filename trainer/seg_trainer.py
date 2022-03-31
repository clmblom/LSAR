import torch
import os
from models.metrics import SegmentationMetric
from torch.utils.tensorboard import SummaryWriter
import utils.patchmaker as pm
import utils.polyfuncs


class Trainer:
    def __init__(self, model, criterion, optimizer, config, data_loader, valid_data_loader=None, device='cpu', scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.train_loader = data_loader
        self.valid_loader = valid_data_loader
        self.dataloaders = {'train': self.train_loader, 'val': self.valid_loader}

        if 'save_from_epoch' in self.config['trainer']:
            self.save_from_epoch = self.config['trainer']['save_from_epoch']
        else:
            self.save_from_epoch = 0
        print(f"Saving from {self.save_from_epoch}")
        self.device = device

        self.epochs = config['trainer']['epochs']
        self.save_per = config['trainer']['save_per']
        self.start_epoch = 1

        self.best_val_metric = 10000000000.0

        self.proj_id = config['name']
        self.save_dir = str(config.save_dir)
        print(f"Tensorboard writing to {f'runs/segmentation/{self.proj_id}'}")

        self.train_batch_size = config['train_dataloader']['batch_size']
        self.val_batch_size = config['val_dataloader']['batch_size']

        self.number_of_accumulated_batches = config['batch_accumulation']
        print(f"Accumulating {self.number_of_accumulated_batches} batches of size {self.train_batch_size}")
        if config.fine_tune:
            self._fine_tune()
        elif config.resume:
            self._load_checkpoint()

    def train_epoch(self):
        self.train(self.start_epoch, self.start_epoch+1)

    def train_model(self):
        self.train(self.start_epoch, self.epochs)

    def train(self, start_epoch, end_epoch):
        print("Start training...")
        seg_metric = SegmentationMetric()
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_metric = {'train': dict(), 'val': dict()}
            running_loss = {'train': [], 'val': []}
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    self.optimizer.zero_grad()
                    for i, (images, masks) in enumerate(self.dataloaders[phase]):
                        with torch.set_grad_enabled(phase == 'train'):

                            images = images.to(self.device)
                            masks = masks.to(self.device)  # shape: (batch, time)

                            output = self.model(images)
                            loss = self.criterion(output, masks)

                            if phase == 'train':
                                loss.backward()
                                if (i + 1) % self.number_of_accumulated_batches == 0 or (i + 1) == len(self.dataloaders[phase]):
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()

                            batch_loss = loss.item() # Since I take batch_mean
                            running_loss[phase] += [batch_loss]

                        if i % 10 == 0 and i > 0:
                            mean_running_loss = sum(running_loss[phase]) / (i + 1)

                            print('{} \t ||e_loss: {:.2f} || b_loss: {:.2f} |||| lr: {:.5f}'.format(
                                i * self.train_batch_size, mean_running_loss,
                                batch_loss,
                                float(self.scheduler.get_last_lr()[0]) if self.scheduler else 0.0))
                    epoch_loss = sum(running_loss[phase]) / len(self.dataloaders[phase])
                    epoch_metric[phase]['Focal loss'] = epoch_loss

                else:
                    self.model.eval()
                    for i, (images, points, patch_size, patch_overlap, sub_batch) in enumerate(self.dataloaders[phase]):
                        with torch.set_grad_enabled(phase == 'train'):
                            patches, wcs, hcs = pm.patchify_tensor(images.squeeze(0),
                                                                   patch_size=patch_size,
                                                                   overlap=patch_overlap)
                            output = torch.zeros((patches.shape[0], 1, patches.shape[2], patches.shape[3]))
                            for sb in range(0, patches.shape[0], sub_batch):
                                patch_sub_batch = patches[sb:sb+sub_batch].to(self.device)
                                out = self.model(patch_sub_batch)
                                output[sb:sb+sub_batch] = out.cpu()
                            patch_sub_batch = patches[sb:].to(self.device)
                            out = self.model(patch_sub_batch)
                            output[sb:] = out.cpu()
                            mask = pm.stitch_mask_tensor(output, wcs, hcs)
                            mask_points = utils.polyfuncs.mask_to_contours(mask)
                            seg_metric.update(points, mask_points)
                    val_metric_dict = seg_metric.calc_metric()
                    epoch_metric[phase] = val_metric_dict
                    epoch_metric[phase]['m'] = epoch_metric[phase]['m1'] * epoch_metric[phase]['m2'] * \
                                               epoch_metric[phase]['m3'] * epoch_metric[phase]['m4']
                    seg_metric.reset()
            self._dump_to_tensorboard(epoch, epoch_metric)
            self._save_checkpoint(epoch, epoch_metric)

    def _fine_tune(self):
        file_name = os.path.join(str(self.config.fine_tune))
        checkpoint = torch.load(file_name)
        self.start_epoch = 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _save_checkpoint(self, epoch, epoch_metric):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_metric': self.best_val_metric
        }
        file_name = os.path.join(self.save_dir,
                                 "checkpoint_epoch_{}_m1_{:.3f}_m2_{:.3f}_m3_{:.3f}_m4_{:.3f}_iou_{:.3f}_m_{:.3f}.pth"
                                 .format(epoch, epoch_metric['val']['m1'], epoch_metric['val']['m2'],
                                         epoch_metric['val']['m3'], epoch_metric['val']['m4'], epoch_metric['val']['iou'],
                                         epoch_metric['val']['m']))
        print("Saving checkpoint to {}".format(file_name))
        torch.save(state, file_name)

    def _load_checkpoint(self):
        file_name = os.path.join(self.save_dir, str(self.config.resume))
        checkpoint = torch.load(file_name)
        self.start_epoch = checkpoint['epoch']+1
        if 'best_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_metric']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _dump_to_tensorboard(self, epoch, metrics):
        print("Dump to tensorboard...")
        print(f"Epoch: {epoch}")
        writer = SummaryWriter(f'runs/segmentation/{self.proj_id}')
        for phase in metrics:
            for metric in metrics[phase]:
                writer.add_scalar(f'{metric}/{phase}', metrics[phase][metric], epoch)

        writer.flush()
        writer.close()
        print("Dump complete")
