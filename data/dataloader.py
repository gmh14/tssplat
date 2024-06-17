from .dataset import Wonder3DImgDataset, MistubaImgDataset, BlenderImgDataset
from utils.config import parse_structured, get_device
from utils.typing import *
import torch
import numpy as np

import sys
import os
import random
from dataclasses import dataclass, field


class DataLoader:
    @dataclass
    class Config:
        batch_size: int
        total_num_iter: int

        world_size: int
        rank: int

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.dataset = None

    def __len__(self):
        return self.n_images

    def to_torch(self):
        img = torch.tensor(np.array(self.dataset.all_tgt_imgs),
                           dtype=torch.float32).to(self.device)
        n = torch.tensor(np.array(self.dataset.all_tgt_ns),
                         dtype=torch.float32).to(self.device)
        d = torch.tensor(np.array(self.dataset.all_tgt_ds),
                         dtype=torch.float32).to(self.device)
        mv = torch.tensor(np.array(self.dataset.all_mv_mats),
                          dtype=torch.float32).to(self.device)
        campos = torch.tensor(np.array(self.dataset.all_campos),
                              dtype=torch.float32).to(self.device)
        mvp = torch.tensor(np.array(self.dataset.all_mvp_mats),
                           dtype=torch.float32).to(self.device)
        bg = torch.tensor(np.array(self.dataset.bgs),
                          dtype=torch.float32).to(self.device)

        iter_res, iter_spp = self.dataset.resolution, self.dataset.spp

        img = torch.cat(
            (torch.lerp(bg, img[..., 0:3], img[..., 3:4]), img[..., 3:4]), dim=-1)

        output = {
            "mv": mv,
            "mvp": mvp,
            "campos": campos,
            "resolution": iter_res,
            "spp": iter_spp,
            "img": img,
            "n": n,
            "d": d,
            "background": bg,
        }
        return output

    def prepare_data(self):
        assert self.dataset is not None

        # convert np to torch
        self.data_all = self.to_torch()

        # split into batches according to world size and rank
        training_data_size = len(self.dataset.all_tgt_imgs)
        num_forward_per_iter = training_data_size // (
            self.cfg.batch_size * self.cfg.world_size)
        if training_data_size % (self.cfg.batch_size * self.cfg.world_size) != 0:
            num_forward_per_iter += 1

        self.num_forward_per_iter = num_forward_per_iter

        appended_training_data_size = num_forward_per_iter * \
            self.cfg.batch_size * self.cfg.world_size
        index_list = list(
            range(appended_training_data_size * self.cfg.total_num_iter))
        index_list = [i % training_data_size for i in index_list]
        # shuffle
        random.seed(1234)
        random.shuffle(index_list)

        self.batch_list = []
        for iter in range(self.cfg.total_num_iter):
            batch_iter = []

            index_list = list(range(training_data_size))
            random.shuffle(index_list)

            for forw_id in range(num_forward_per_iter):
                batch_idx_forw_id = []

                for rank_i in range(self.cfg.world_size):
                    start = rank_i * self.cfg.batch_size
                    end = min(start + self.cfg.batch_size, training_data_size)
                    # print(f"({start},{end})")
                    batch_idx_forw_id.append(index_list[start:end])

                batch_iter.append(batch_idx_forw_id)
            self.batch_list.append(batch_iter)

    def __call__(self, iter, forward_id):
        batch_ids = self.batch_list[iter][forward_id][self.cfg.rank]
        output = {
            "mv": self.data_all["mv"][batch_ids],
            "mvp": self.data_all["mvp"][batch_ids],
            "campos": self.data_all["campos"][batch_ids],
            "resolution": self.data_all["resolution"],
            "spp": self.data_all["spp"],
            "img": self.data_all["img"][batch_ids],
            "background": self.data_all["background"][batch_ids],
            "n": self.data_all["n"][batch_ids],
            "d": self.data_all["d"][batch_ids],
        }
        return output


class Wonder3DDataLoader(DataLoader):
    @dataclass
    class Config(DataLoader.Config):
        dataset_config: Wonder3DImgDataset.Config

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.dataset = Wonder3DImgDataset(self.cfg.dataset_config)

        self.prepare_data()


class MistubaImgDataLoader(DataLoader):
    @dataclass
    class Config(DataLoader.Config):
        dataset_config: MistubaImgDataset.Config

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.dataset = MistubaImgDataset(self.cfg.dataset_config)

        self.prepare_data()


class BlenderImgDataLoader(DataLoader):
    @dataclass
    class Config(DataLoader.Config):
        dataset_config: BlenderImgDataset.Config

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.dataset = BlenderImgDataset(self.cfg.dataset_config)

        self.prepare_data()
