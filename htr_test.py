import argparse
from datasets.htr import deterministic_set
from data_loaders.base_data_loader import BaseDataLoader
from parse_config import ConfigParser
from models.htr_transformer import transformer_line
from trainer.htr_tester import Tester
from tokenizers import Tokenizer
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
    dataset_dict = {"DeterministicSet": deterministic_set}
    tokenizer = Tokenizer.from_file(config['tokenizer'])

    test_datasets = []
    for dataset in config['test_datasets']:
        name = dataset['name']
        module = dataset_dict[name]
        m = getattr(module, name)
        test_datasets.append(m(**dataset['args'], tokenizer=tokenizer))

    test_data = test_datasets[0]

    print("Test data:", len(test_data))

    test_loader = BaseDataLoader(test_data,
                                 batch_size=config['test_dataloader']['batch_size'],
                                 shuffle=config['test_dataloader']['shuffle'],
                                 num_workers=config['test_dataloader']['num_workers'],
                                 drop_last=config['test_dataloader']['drop_last'])
    model = transformer_line(hidden_size=config['model']['encoder']['hidden_size'],
                             ex_feature_size=config['model']['feature_extractor']['features'],
                             ex_feature_height=test_datasets[0].im_size[1]//config['model']['feature_extractor']['height_div'],
                             ex_feature_width=test_datasets[0].im_size[0]//config['model']['feature_extractor']['width_div'],
                             tar_len=test_datasets[0].pad_text+1,
                             output_size=tokenizer.get_vocab_size(with_added_tokens=True),
                             num_dec_layers=config['model']['decoder'].get('num_layers', 6),
                             dropout_p=config['model']['decoder'].get('dropout_p', 0.5),
                             norm_first=config['model']['decoder'].get('norm_first', False))
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
