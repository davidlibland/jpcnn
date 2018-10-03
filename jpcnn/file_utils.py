import json
import pathlib
from datetime import datetime
import os
from dataclasses import asdict


def build_checkpoint_dir_name():
    """Builds a dir name"""
    return datetime.now().strftime("Checkpoint-%Y%m%d-%H%M%S")


def build_checkpoint_file_name(dir, descriptor):
    """Builds a relative filename to save the checkpoint at"""
    return "./{}/params_{}.ckpt".format(dir, descriptor)


def load_or_save_conf(ckpt_file, conf):
    from jpcnn.config import JPCNNConfig
    if ckpt_file is not None:
        dir_name = os.path.dirname(ckpt_file)
        with open("%s/conf.json" % dir_name, "r") as fp:
            conf = JPCNNConfig(**json.load(fp))
    else:
        dir_name = build_checkpoint_dir_name()
        root_dir = os.path.join(os.getcwd(), dir_name)
        pathlib.Path(root_dir).mkdir(exist_ok = True)
        with open(os.path.join(os.getcwd(), dir_name, "conf.json"), "w") as fp:
            json.dump(asdict(conf), fp)
    return conf, dir_name
