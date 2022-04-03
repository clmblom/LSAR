import argparse
from datasets.segmentation import coco_patch_validation_set
from data_loaders.base_data_loader import BaseDataLoader
from parse_config import ConfigParser
from models.fpn_segmenter import fpn_segmenter
from trainer.seg_tester import Tester
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
    dataset_dict = {"CocoPatchValidationSet": coco_patch_validation_set}

    test_datasets = []
    for dataset in config['test_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        test_datasets.append(m(**dataset['args']))

    test_data = test_datasets[0]

    print("Test data:", len(test_data))

    test_loader = BaseDataLoader(test_data,
                                 batch_size=config['test_dataloader']['batch_size'],
                                 shuffle=config['test_dataloader']['shuffle'],
                                 num_workers=config['test_dataloader']['num_workers'],
                                 drop_last=config['test_dataloader']['drop_last'])
    model = fpn_segmenter(backbone_feature_size=config['model']['backbone']['features'])
    model = model.to(device)
    tester = Tester(model=model,
                    config=config,
                    test_data_loader=test_loader,
                    device=device)
    tester.test_model()


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
