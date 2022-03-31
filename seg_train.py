from datasets.segmentation import alto_crop_train_set, coco_patch_validation_set
from models.fpn_segmenter import fpn_segmenter
from models.losses import FocalTverskyLoss
from torch.optim import AdamW
import torch
from parse_config import ConfigParser
from data_loaders.base_data_loader import BaseDataLoader
from trainer.seg_trainer import Trainer
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print(device)
else:
    print(torch.cuda.get_device_name(0))
print("Cpu threads: {}".format(torch.get_num_threads()))


def main(config):
    dataset_dict = {"AltoCropTrainSet": alto_crop_train_set, "CocoPatchValidationSet": coco_patch_validation_set}
    train_datasets = []
    for dataset in config['train_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        train_datasets.append(m(**dataset['args']))

    val_datasets = []
    for dataset in config['val_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        val_datasets.append(m(**dataset['args']))

    if not val_datasets:
        assert val_datasets, "Add a validation set"
    else:
        train_data = train_datasets[0]
        val_data = val_datasets[0]

    print("Training data:", len(train_data), "Validation data:", len(val_data))

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
    model = fpn_segmenter(backbone_feature_size=config['model']['backbone']['features'])
    model = model.to(device)
    criterion = FocalTverskyLoss(gamma=2)
    optimizer = AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
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
