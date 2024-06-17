
import torch
import numpy as np
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass, field

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import parse_structured  # NOQA
from utils.typing import *  # NOQA


class Wonder3DImgDataset:
    @dataclass
    class Config:
        camera_mvp_root: str
        camera_views: List[str]
        image_root: str

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        # load image data
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # background as all white
        self.bgs = [np.ones((self.all_tgt_imgs[0].shape[0], self.all_tgt_imgs[0].shape[1], 3))
                    for i in range(len(self.all_tgt_imgs))]

        # camera params
        self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(
            self.all_mv_mats[0])
        self.camera_dist = np.linalg.norm(self.all_campos[0])

        # resolution and spp
        self.resolution = self.all_tgt_imgs[0].shape[0]  # square imgs
        self.spp = 1

    def load_img_data(self):
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        for view in self.cfg.camera_views:
            camera_filename = f"{self.cfg.camera_mvp_root}/{view}_mvp.npy"
            all_mvp_mats.append(np.load(camera_filename))
            all_tgt_imgs.append(np.zeros(1))
            all_tgt_ns.append(np.zeros(1))

        # masked images
        img_root = os.path.dirname(self.cfg.image_root) + "/masked_colors1"
        for img_file in os.listdir(img_root):
            found = False
            for i, c in enumerate(self.cfg.camera_views):
                if c in img_file:
                    found = True
                    tgt_img = np.array(Image.open(os.path.join(
                        img_root, img_file))).astype(np.float32) / 255.0

                    tgt_img = cv2.resize(tgt_img, (512, 512), cv2.INTER_CUBIC)

                    # print(tgt_img.shape)
                    tgt_img[..., 3] = np.where(tgt_img[..., 3] < 0.8, 0, 1)
                    # plt.imshow(tgt_img)
                    # plt.show()

                    all_tgt_imgs[i] = tgt_img
                    break

            assert found

        # normal images
        img_root = os.path.dirname(self.cfg.image_root) + "/normals"
        for img_file in os.listdir(img_root):
            found = False
            for i, c in enumerate(self.cfg.camera_views):
                if c in img_file:
                    found = True
                    tgt_img = np.array(Image.open(os.path.join(
                        img_root, img_file))).astype(np.float32) / 255.0

                    tgt_img = cv2.resize(tgt_img, (512, 512), cv2.INTER_CUBIC)

                    # print(tgt_img.shape)
                    tgt_img[..., 0:3] = (tgt_img[..., 0:3] - 0.5) * 2

                    all_tgt_ns[i] = tgt_img
                    break

            assert found

        imgs = []
        mvps = []
        ns = []
        ds = []
        for i, (mvp, img, n) in enumerate(zip(all_mvp_mats, all_tgt_imgs, all_tgt_ns)):
            if len(img.shape) == 3:
                imgs.append(img)
                ds.append(img[..., -1:])
                ns.append(n)
                mvps.append(mvp.astype(np.float32))
                all_campos.append(np.asarray([0, 0, 1]))

        all_tgt_imgs = imgs
        all_tgt_ns = ns
        all_tgt_ds = ds
        all_mvp_mats = mvps
        all_mv_mats = all_mvp_mats
        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds


class MistubaImgDataset:
    @dataclass
    class Config:
        image_root: str

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        # load image data
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # background as all white
        self.bgs = [np.ones((self.all_tgt_imgs[0].shape[0], self.all_tgt_imgs[0].shape[1], 3))
                    for i in range(len(self.all_tgt_imgs))]

        # camera params
        self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(
            self.all_mv_mats[0])
        self.camera_dist = np.linalg.norm(self.all_campos[0])

        # resolution and spp
        self.resolution = self.all_tgt_imgs[0].shape[0]  # square imgs
        self.spp = 1

    def load_img_data(self):
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        assert os.path.isdir(self.cfg.image_root)
        tgt_path = self.cfg.image_root
        for img_file in glob.glob(os.path.join(tgt_path, "img*rgba*.png")):
            # print(img_file)
            tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            all_tgt_imgs.append(tgt_img)

            img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]

            mvp_mat_file = os.path.join(
                tgt_path, "mvp_mtx_{}.npy".format(img_id))
            mvp_mat = np.load(mvp_mat_file)
            all_mvp_mats.append(mvp_mat)

            mv_file = os.path.join(tgt_path, "mv_{}.npy".format(img_id))
            mv = np.load(mv_file)
            all_mv_mats.append(mv)

            campos = np.linalg.inv(mv)[:3, 3]
            all_campos.append(campos)

            normal_file = os.path.join(
                tgt_path, "normal_{}.npy".format(img_id))
            if os.path.exists(normal_file):
                n = np.load(normal_file)
            else:
                n = np.zeros_like(tgt_img)
            all_tgt_ns.append(n)

            depth_file = os.path.join(tgt_path, "depth_{}.npy".format(img_id))
            if os.path.exists(depth_file):
                d = np.load(depth_file)
                d = d[..., None]
            else:
                d = np.zeros_like(tgt_img)
            all_tgt_ds.append(d)
            # print(f"d:{np.max(d)}, {np.min(d)}")

            try:
                assert np.all(np.isfinite(tgt_img))
                assert np.all(np.isfinite(mvp_mat))
                assert np.all(np.isfinite(mv))
                assert np.all(np.isfinite(campos))
                # assert np.all(np.isfinite(n))
                assert np.all(np.isfinite(d))
            except:
                import pdb
                pdb.set_trace()

        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds


class BlenderImgDataset:
    @dataclass
    class Config:
        image_root: str

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        self.cfg = parse_structured(self.Config, cfg)
        # load image data
        self.all_tgt_imgs, self.all_mvp_mats, self.all_mv_mats, self.all_campos, self.all_tgt_ns, self.all_tgt_ds = self.load_img_data()

        # background as all white
        self.bgs = [np.ones((self.all_tgt_imgs[0].shape[0], self.all_tgt_imgs[0].shape[1], 3))
                    for i in range(len(self.all_tgt_imgs))]

        # camera params
        self.camera_p = self.all_mvp_mats[0] @ np.linalg.inv(
            self.all_mv_mats[0])
        self.camera_dist = np.linalg.norm(self.all_campos[0])

        # resolution and spp
        self.resolution = self.all_tgt_imgs[0].shape[0]  # square imgs
        self.spp = 1

    def load_img_data(self):
        all_tgt_imgs = []
        all_mvp_mats = []
        all_mv_mats = []
        all_campos = []
        all_tgt_ns = []
        all_tgt_ds = []

        assert os.path.isdir(self.cfg.image_root)
        tgt_path = self.cfg.image_root
        for img_file in glob.glob(os.path.join(tgt_path, "img*rgba*.png")):
            # print(img_file)
            tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            all_tgt_imgs.append(tgt_img)

            img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]

            mvp_mat_file = os.path.join(
                tgt_path, "mvp_mtx_{}.npy".format(img_id))
            mvp_mat = np.load(mvp_mat_file)
            all_mvp_mats.append(mvp_mat)

            mv_file = os.path.join(tgt_path, "mv_{}.npy".format(img_id))
            mv = np.load(mv_file)
            all_mv_mats.append(mv)

            campos = np.linalg.inv(mv)[:3, 3]
            all_campos.append(campos)

            ##### TODO
            n = np.zeros_like(tgt_img)
            d = np.zeros_like(tgt_img)
            # all_tgt_ns.append(n)
            # all_tgt_ds.append(d)
            normal_file = os.path.join(
                tgt_path, "normal_{}.npy".format(img_id))
            if os.path.exists(normal_file):
                n = np.load(normal_file)
            else:
                n = np.zeros_like(tgt_img)
            all_tgt_ns.append(n)

            depth_file = os.path.join(tgt_path, "depth_{}.npy".format(img_id))
            if os.path.exists(depth_file):
                d = np.load(depth_file)
                d = d[..., None]
            else:
                d = np.zeros_like(tgt_img)
            all_tgt_ds.append(d)
            print(f"d:{np.max(d)}, {np.min(d)}")

            try:
                assert np.all(np.isfinite(tgt_img))
                assert np.all(np.isfinite(mvp_mat))
                assert np.all(np.isfinite(mv))
                assert np.all(np.isfinite(campos))
                # assert np.all(np.isfinite(n))
                assert np.all(np.isfinite(d))
            except:
                import pdb
                pdb.set_trace()

        return all_tgt_imgs, all_mvp_mats, all_mv_mats, all_campos, all_tgt_ns, all_tgt_ds
