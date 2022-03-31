from utils.util import read_json
from pathlib import Path


class ConfigParser:
    def __init__(self, config, resume=None, exp_id=None, fine_tune=None):
        self.resume = resume
        self.exp_id = exp_id
        self.config = config
        self.fine_tune = fine_tune

        save_dir = Path(config['trainer']['save_dir'])
        proj_id = config['name']
        self.save_dir = save_dir / proj_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def read_args(cls, args):
        args = args.parse_args()

        if args.config:
            config_filename = args.config
        else:
            msg_no_config = "Add a config file to arguments"
            assert args.config is not None, msg_no_config

        config = read_json(config_filename)

        if args.fine_tune:
            fine_tune = args.fine_tune
        else:
            fine_tune = None
        if args.resume:
            resume = Path(args.resume)
        else:
            resume = None
        return cls(config=config, resume=resume, fine_tune=fine_tune)

    def __getitem__(self, item):
        return self.config[item]
