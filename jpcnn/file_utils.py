import json
import pathlib
from datetime import datetime
import os
from dataclasses import asdict
from jpcnn.dropbox_utils import setup_dropbox_syncs


def build_checkpoint_dir_name():
    """Builds a dir name"""
    return datetime.now().strftime("Checkpoint-%Y%m%d-%H%M%S")


def build_checkpoint_file_name(dir, descriptor):
    """Builds a relative filename to save the checkpoint at"""
    return "./{}/params_{}.ckpt".format(dir, descriptor)


def load_or_save_conf(ckpt_file, conf, access_token=None):
    from jpcnn.config import JPCNNConfig
    do_down_sync = False
    if ckpt_file is not None:
        dir_name = os.path.dirname(ckpt_file)
        try:
            with open("%s/conf.json" % dir_name, "r") as fp:
                conf = JPCNNConfig(**json.load(fp))
        except FileNotFoundError:
            do_down_sync = True
    else:
        dir_name = build_checkpoint_dir_name()
        root_dir = os.path.join(os.getcwd(), dir_name)
        pathlib.Path(root_dir).mkdir(exist_ok = True)
        with open(os.path.join(os.getcwd(), dir_name, "conf.json"), "w") as fp:
            json.dump(asdict(conf), fp)
    if not access_token:
        access_token = input("What is your dropbox access token?")
    if access_token:
        down_sync, up_sync = setup_dropbox_syncs(
            access_token = access_token,
            dropbox_root = conf.dropbox_root,
            local_root_dir = "./",
            dir_name = dir_name
        )
        if do_down_sync:
            down_sync()
        else:
            up_sync()
    elif do_down_sync:
        raise ValueError("File and access token are missing.")
    else:
        up_sync = lambda: None
    return conf, dir_name, up_sync
