import argparse
from datasets.htr import deterministic_set, streaming_set
from data_loaders.base_data_loader import BaseDataLoader
from parse_config import ConfigParser
from models.htr_transformer import transformer_line
from trainer.htr_trainer import Trainer
from tokenizers import Tokenizer

import torch.optim as optim
import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print(device)
else:
    print(torch.cuda.get_device_name(0))
print("Cpu threads: {}".format(torch.get_num_threads()))


"""
File setup is heavily inspired by https://github.com/victoresque/pytorch-template
"""


def main(config):
    dataset_dict = {"DeterministicSet": deterministic_set, "StreamingSet": streaming_set}
    if "OMOSet" in config['train_datasets'][0]['name']:
        from datasets.htr import omo_set
        dataset_dict["OMOSet"] = omo_set
    tokenizer = Tokenizer.from_file(config['tokenizer'])

    train_datasets = []
    for dataset in config['train_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        train_datasets.append(m(**dataset['args'], tokenizer=tokenizer))

    val_datasets = []
    for dataset in config['val_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        val_datasets.append(m(**dataset['args'], tokenizer=tokenizer))

    """"""
    if not val_datasets:
        assert val_datasets, "Add a validation set"
    else:
        train_data = train_datasets[0]
        val_data = val_datasets[0]

    print("Train data:", len(train_data), "Val data:", len(val_data))

    train_loader = BaseDataLoader(train_data,
                                  batch_size=config['train_dataloader']['batch_size'],
                                  shuffle=config['train_dataloader']['shuffle'],
                                  num_workers=config['train_dataloader']['num_workers'],
                                  drop_last=config['train_dataloader']['drop_last'])
    val_loader = BaseDataLoader(val_data,
                                batch_size=config['val_dataloader']['batch_size'],
                                shuffle=config['val_dataloader']['shuffle'],
                                num_workers=config['val_dataloader']['num_workers'],
                                drop_last=config['val_dataloader']['drop_last'])
    model = transformer_line(hidden_size=config['model']['encoder']['hidden_size'],
                             ex_feature_size=config['model']['feature_extractor']['features'],
                             ex_feature_height=train_datasets[0].im_size[1]//config['model']['feature_extractor']['height_div'],
                             ex_feature_width=train_datasets[0].im_size[0]//config['model']['feature_extractor']['width_div'],
                             tar_len=train_datasets[0].pad_text+1,
                             output_size=tokenizer.get_vocab_size(with_added_tokens=True),
                             num_dec_layers=config['model']['decoder'].get('num_layers', 6),
                             dropout_p=config['model']['decoder'].get('dropout_p', 0.5),
                             norm_first=config['model']['decoder'].get('norm_first', False))
    model = model.to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    if True:
        print("Using Lambda lr scheduler...")

        class lr_scheduler(object):
            def __init__(self, warm_up):
                self.warm_up = warm_up

            def __call__(self, epoch):
                if self.warm_up == 0:
                    return 1
                elif epoch < self.warm_up and self.warm_up > 0:
                    return min(max(0, (epoch + 1) / self.warm_up), 1)
                else:
                    return (self.warm_up / (epoch + 1)) ** 0.5
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler(0), verbose=True)
    else:
        scheduler = None

    train = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    config=config,
                    data_loader=train_loader,
                    valid_data_loader=val_loader,
                    device=device,
                    scheduler=scheduler)
    train.train_model()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Stuff')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-f', '--fine_tune', default=None, type=str,
                      help='path to .pth file  (default: None)')
    config = ConfigParser.read_args(args)
    main(config)
