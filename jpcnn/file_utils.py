import json
import pathlib
from datetime import datetime
import re
import os

from dataclasses import asdict

from jpcnn.train import JPCNNConfig

epoch_re = re.compile(r"^params_\w*_(?P<epoch>[0-9]*).")

def build_checkpoint_dir_name():
    """Builds a dir name"""
    return datetime.now().strftime("Checkpoint-%Y%m%d-%H%M%S")

def build_checkpoint_file_name(dir, descriptor, epoch):
    """Builds a relative filename to save the checkpoint at"""
    return "./{}/params_{}_{}.ckpt".format(dir, descriptor, epoch)

def parse_checkpoint_file_name(fname):
    """Parses a filename for the epoch and dir name"""
    base_name = os.path.basename(fname)
    dir_name = os.path.dirname(fname)
    return int(epoch_re.match(base_name)["epoch"]), dir_name


def load_or_save_conf(ckpt_file, conf):
    if ckpt_file is not None:
        start_epoch, dir_name = parse_checkpoint_file_name(ckpt_file)
        with open("%s/conf.json" % dir_name, "r") as fp:
            conf = JPCNNConfig(**json.load(fp))
    else:
        dir_name = build_checkpoint_dir_name()
        start_epoch = 0
        root_dir = os.path.join(os.getcwd(), dir_name)
        pathlib.Path(root_dir).mkdir(exist_ok = True)
        with open(os.path.join(os.getcwd(), dir_name, "conf.json"), "w") as fp:
            json.dump(asdict(conf), fp)
    return conf, dir_name, start_epoch