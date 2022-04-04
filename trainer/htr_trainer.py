import torch
import torch.nn.functional as F
import os
import models.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

import utils.util


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

        self.save_from_epoch = self.config['trainer']['save_from_epoch']

        print(f"Saving from {self.save_from_epoch}")
        self.device = device

        self.epochs = config['trainer']['epochs']
        self.save_per = config['trainer']['save_per']
        self.start_epoch = 1

        self.best_val_metric = 10000000000.0

        self.proj_id = config['name']
        self.save_dir = str(config.save_dir)
        print(f"Tensorboard writing to {f'runs/htr/{self.proj_id}'}")

        self.pad_text = config['train_datasets'][0]['args']['pad_text']
        self.train_batch_size = config['train_dataloader']['batch_size']
        self.val_batch_size = config['val_dataloader']['batch_size']

        self.number_of_accumulated_batches = config['batch_accumulation']
        print(f"Accumulating {self.number_of_accumulated_batches} batches of size {self.train_batch_size}")

        self.deterministic = config['train_datasets'][0]['args']['deterministic']
        self.augment = config['train_datasets'][0]['args']['augment']
        print("Deterministic sets:", self.deterministic)
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
        label_smoothing = True
        lm = metrics.LosenMetric()
        batch_lm = metrics.LosenMetric()

        for epoch in range(start_epoch, end_epoch+1):
            epoch_metric = {'train': {'Loss': 0, 'CER': 0, 'CLS': 0},
                            'val': {'Loss': 0, 'CER': 0, 'CLS': 0}}

            for phase in ['train', 'val']:
                lm.reset()
                if phase == 'train':
                    self.model.train()
                    tokenizer = self.dataloaders['train'].dataset.tokenizer
                else:
                    self.model.eval()
                    tokenizer = self.dataloaders['val'].dataset.tokenizer

                self.optimizer.zero_grad()
                for i, (images, words, image_padding, word_padding) in enumerate(self.dataloaders[phase]):
                    with torch.set_grad_enabled(phase == 'train'):

                        images = images.to(self.device)
                        # words are <S>the_word<E>
                        words = words.to(self.device)  # shape: (batch, time)

                        words_hot = F.one_hot(words, num_classes=self.model.decoder.output_size).float().squeeze(2)  # shape: batch, time, output_size
                        words = words.transpose(1, 0)  # (batch, time) -> (time, batch)

                        # Send in <S>the_word
                        output, _, _, _ = self.model(images, target=words[:-1, :])  # shape: time, batch, output-size
                        predictions = torch.argmax(output, dim=-1)  # time, batch
                        output = output.transpose(1, 0)  # shape: batch, time, output_size

                        if label_smoothing:
                            alpha = 0.4
                            k = self.model.decoder.output_size
                            words_hot = words_hot * alpha + (1 - alpha) / k

                        # expect the_word<E> as output
                        loss_enc_dec = self.criterion(F.log_softmax(output, dim=2), words_hot[:, 1:, :])

                        if phase == 'train':
                            loss_enc_dec.backward()
                            if (i+1) % self.number_of_accumulated_batches == 0 or (i+1) == len(self.dataloaders[phase]):
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                        batch_loss = loss_enc_dec.item() # Since I take batch_mean
                        epoch_metric[phase]['Loss'] += batch_loss

                        words = words.cpu()
                        predictions = predictions.cpu()

                        skip_tokens = [tokenizer.token_to_id(t) for t in ['<S>', '<E>', '<P>']]
                        targets = [''.join([tokenizer.id_to_token(t) for t in token if t not in skip_tokens]) for token in words.T]
                        predictions = [''.join([tokenizer.id_to_token(t) for t in token if t not in skip_tokens]) for token in predictions.T]
                        lm.update(targets, predictions)

                    if i % 100 == 0 and i > 0:
                        batch_lm.update(targets, predictions)
                        batch_cer, batch_cls_cer = batch_lm.calc_metric()
                        batch_lm.reset()

                        running_loss = epoch_metric[phase]['Loss']/(i+1)
                        running_cer, running_cls = lm.calc_metric()

                        print('{} \t ||e_loss: {:.2f} | e_cer: {:.2f} | e_cls: {:.2f}|| b_loss: {:.2f} | b_cer: {:.2f} | cls_cer: {:.2f}|||| lr: {:.5f}'.format(
                            i*(self.train_batch_size if phase == 'train' else self.val_batch_size), running_loss,
                            running_cer, running_cls, batch_loss, batch_cer, batch_cls_cer,
                            float(self.scheduler.get_last_lr()[0]) if self.scheduler else 0.0))
                epoch_loss = epoch_metric[phase]['Loss'] / len(self.dataloaders[phase])
                epoch_cer, epoch_cls = lm.calc_metric()
                epoch_metric[phase]['CER'] = epoch_cer
                epoch_metric[phase]['CLS'] = epoch_cls
                epoch_metric[phase]['Loss'] = epoch_loss

                print('{}:{} Loss: {:.2f} CER: {:.2f}'.format(
                    epoch, phase, epoch_loss, epoch_cer, epoch_cls))
                """
                if phase == 'val' and (epoch % self.save_per == 0 or epoch_cer < self.best_val_metric) and epoch >= self.save_from_epoch:
                    if epoch_cer < self.best_val_metric:
                        self.best_val_metric = epoch_cer
                    self._save_checkpoint(epoch, epoch_cer)
                """

            self._dump_to_tensorboard(epoch, epoch_metric)
            if (epoch % self.save_per == 0 or epoch_cer < self.best_val_metric) and epoch >= self.save_from_epoch:
                if epoch_metric['val']['CER'] < self.best_val_metric:
                    self.best_val_metric = epoch_metric['val']['CER']
            if self.scheduler:
                self.scheduler.step()
            self._save_checkpoint(epoch, epoch_metric['val']['CER'])
        return

    def _fine_tune(self):
        file_name = os.path.join(str(self.config.fine_tune))
        checkpoint = torch.load(file_name)
        self.start_epoch = 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("Resetting Lambda lr scheduler...")
            print("Warm-up until epoch", self.config['trainer']['warm_up_to'])
            self.scheduler = self.scheduler.__class__(self.optimizer,
                                                      lr_lambda=utils.util.LrScheduler(self.config['trainer']['warm_up']),
                                                      verbose=True)

    def _save_checkpoint(self, epoch, running_cer):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_metric': self.best_val_metric
        }
        file_name = os.path.join(self.save_dir, "checkpoint_epoch_{}_cer_{:.2f}.pth".format(epoch, running_cer))
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

    def _dump_to_tensorboard(self, epoch, metrics_dict):
        print("Dump to tensorboard...")
        print(f"Epoch: {epoch}")
        writer = SummaryWriter(f'runs/htr/{self.proj_id}')
        for phase in metrics_dict:
            for metric in metrics_dict[phase]:
                writer.add_scalar(f'{metric}/{phase}', metrics_dict[phase][metric], epoch)

        writer.flush()
        writer.close()
        print("Dump complete")
