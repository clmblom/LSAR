from datasets.segmentation import inference_patch_set
from models.fpn_segmenter import fpn_segmenter
import torch
from parse_config import ConfigParser
from data_loaders.base_data_loader import BaseDataLoader
from trainer.seg_predicter import Predicter
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print(device)
else:
    print(torch.cuda.get_device_name(0))
print("Cpu threads: {}".format(torch.get_num_threads()))


def main(config):
    dataset_dict = {"InferencePatchSet": inference_patch_set}
    inf_datasets = []
    for dataset in config['inf_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        inf_datasets.append(m(**dataset['args']))
    inf_data = inf_datasets[0]

    print("Inference data:", len(inf_data))
    inf_loader = BaseDataLoader(inf_data,
                                batch_size=1,
                                shuffle=config['inf_dataloader']['shuffle'],
                                num_workers=config['inf_dataloader']['num_workers'],
                                drop_last=config['inf_dataloader']['drop_last'])
    model = fpn_segmenter(backbone_feature_size=config['model']['backbone']['features'])
    model = model.to(device)
    predicter = Predicter(model=model,
                          config=config,
                          inf_data_loader=inf_loader,
                          device=device)
    predicter.predict()


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
