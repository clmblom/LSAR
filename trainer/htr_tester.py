import torch
import os
import models.metrics as metrics


class Tester:
    def __init__(self, model, config, test_data_loader, device='cpu'):
        self.model = model

        self.config = config

        self.test_loader = test_data_loader
        self.dataloaders = {'test': self.test_loader}
        self.device = device

        self.proj_id = config['name']
        self.save_dir = str(config.save_dir)

        self.pad_text = config['test_datasets'][0]['args']['pad_text']
        self.batch_size = config['test_dataloader']['batch_size']

        if config.resume:
            self._load_checkpoint()
        else:
            assert config.resume, "No model"

    def test_model(self):
        self.test(1, 1)

    def test(self, start_epoch, end_epoch):
        print("Start testing")
        lm = metrics.LosenMetric()
        batch_lm = metrics.LosenMetric()

        for epoch in range(start_epoch, end_epoch+1):
            epoch_metric = {'test': {'CER': 0, 'CLS': 0}}

            for phase in ['test']:
                lm.reset()
                self.model.eval()
                tokenizer = self.dataloaders['test'].dataset.tokenizer
                for i, (images, words, _, _) in enumerate(self.dataloaders[phase]):
                    print("Batches:", f"{i+1}/{len(self.dataloaders[phase])}")
                    with torch.no_grad():

                        images = images.to(self.device)
                        # words are <S>the_word<E>
                        words = words.to(self.device)  # shape: (batch, time)
                        words = words.transpose(1, 0)  # (batch, time) -> (time, batch)

                        # Send in <S>the_word
                        output, _, _, _ = self.model(images, target=None)  # shape: time, batch, output-size
                        predictions = torch.argmax(output, dim=-1)  # time, batch

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

                        running_cer, running_cls = lm.calc_metric()

                        print('{} \t || e_cer: {:.2f} | e_cls: {:.2f}|| b_cer: {:.2f} | cls_cer: {:.2f}||||'.format(
                            i*self.batch_size,
                            running_cer, running_cls, batch_cer, batch_cls_cer))
                epoch_cer, epoch_cls = lm.calc_metric()
                epoch_metric[phase]['CER'] = epoch_cer
                epoch_metric[phase]['CLS'] = epoch_cls

                print('{}:{} CER {:.2f} CLS_CER: {:.2f}'.format(
                    epoch, phase, epoch_cer, epoch_cls))
            print()
        return

    def _load_checkpoint(self):
        file_name = os.path.join(self.save_dir, str(self.config.resume))
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['state_dict'])