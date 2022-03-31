from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, drop_last, sampler=None):
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': True,
            'sampler': sampler
        }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)
