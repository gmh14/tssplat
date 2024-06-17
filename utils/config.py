import os
from dataclasses import dataclass, field
from datetime import datetime
from omegaconf import OmegaConf, open_dict
import time

import torch

from .typing import *


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


class PrintExecTime:
    def __init__(self, is_print_exec_time):
        self.is_print_exec_time = is_print_exec_time
        self.time_table = dict()

    def __call__(self, string, time):
        if self.is_print_exec_time:
            print(f"{string}: {time:.6f}s")

            if string in self.time_table.keys():
                self.time_table[string].append(time)
            else:
                self.time_table[string] = [time]
        else:
            pass


@dataclass
class Config():
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

        self.print_exec_time = PrintExecTime(dictionary["print_exec_time"])
        self.benchmark_it = dictionary["print_exec_time"]
        self.timestamp_stack = []

        self.set_timestamp()

    def set_timestamp(self):
        if self.benchmark_it:
            self.timestamp_stack.append(time.time())

    def timestamp(self, string: str):
        if self.benchmark_it:
            tcur = time.time()
            self.print_exec_time(string, tcur - self.timestamp_stack[-1])
            self.timestamp_stack[-1] = tcur

    def pop_timestamp(self):
        if self.benchmark_it:
            self.timestamp_stack.pop()

    def __call__(self, string, time):
        self.print_exec_time(string, time)
