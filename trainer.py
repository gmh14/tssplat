import pypgo  # NOQA
import sys
import os
from tqdm import trange, tqdm
import argparse
import numpy as np
from PIL import Image
import torch

from geometry import load_geometry
from materials import load_material
from renderers import MeshRasterizer
from data import load_dataloader
from utils.config import load_config
from utils.optimizer import AdamUniform


class LinearInterpolateScheduler:
    def __init__(self, start_iter, end_iter, start_val, end_val, freq):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.end_val = end_val
        self.freq = freq

    def __call__(self, iter):
        if iter < self.start_iter or iter % self.freq != 0 or iter == 0:
            return None

        p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_val * (1 - p) + self.end_val * p


def train(cfg):
    verbose = cfg.get("verbose", False)
    
    material = None
    cfg.geometry.optimize_geo = True
    cfg.geometry.output_path = cfg.output_path
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)

    shade_loss = torch.nn.MSELoss()

    if cfg.get("fitting_stage", None) == "texture":
        assert cfg.get("material", None) is not None
        material = load_material(cfg.material_type)(cfg.material)
        cfg.geometry.optimize_geo = False
        shade_loss = torch.nn.L1Loss()

    geometry = load_geometry(cfg.geometry_type)(cfg.geometry)
    renderer = MeshRasterizer(geometry, material, cfg.renderer)
    dataloader = load_dataloader(cfg.dataloader_type)(cfg.data)

    num_forward_per_iter = dataloader.num_forward_per_iter

    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.total_num_iter * num_forward_per_iter, eta_min=1e-4)

    permute_surface_scheduler = None
    if cfg.use_permute_surface_v:
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # main loop
    best_loss = 1e10
    best_loss_iter = 0
    best_opt_imgs = None
    best_v = None

    for it in trange(cfg.total_num_iter):
        for forw_id in range(num_forward_per_iter):
            batch = dataloader(it, forw_id)

            color_ref = batch["img"]

            fit_depth = cfg.get("fit_depth", False)
            if fit_depth:
                fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            renderer_input = {
                "mvp": batch["mvp"],
                # "only_alpha": cfg.get("fitting_stage", None) == "geometry",
                "only_alpha": cfg.get("fitting_stage", None) == "geometry",
                "iter_num": it,
                "resolution": batch["resolution"],
                "background": batch["background"],
                "permute_surface_scheduler": permute_surface_scheduler,
                "fit_depth": fit_depth,
                "campos": batch["campos"],
            }

            # forward
            out = renderer(**renderer_input)

            #### mask
            img_loss = None
            if cfg.get("fitting_stage", None) == "geometry":
                img_loss = shade_loss(out["shaded"][..., -1],
                                      color_ref[..., -1])
            else:
                img_loss = shade_loss(
                    out["shaded"][..., :3], color_ref[..., :3])
            img_loss *= 20
            
            #### depth
            if fit_depth:
                img_loss += shade_loss(out["d"][..., -1] * color_ref[..., -1],
                                       batch["d"][..., -1] * color_ref[..., -1]) * 100

            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"]

            loss = img_loss * 100 + reg_loss

            if True:  # it % 100 == 0:
                tqdm.write(
                    "iter=%4d, img_loss=%.4f, reg_loss=%.4f"
                    % (
                        it,
                        img_loss,
                        reg_loss,
                    )
                )

            # backward
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()
            scheduler.step()

            # logging
            cur_loss = loss.clone().detach().cpu().item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_loss_iter = it
                best_v = geometry.tet_v.clone().detach()
                best_opt_imgs = out["shaded"].clone().detach()

            if it % 100 == 0 and forw_id == 0:
                # if False:
                os.makedirs(f"{cfg.output_path}/mesh{it:05d}", exist_ok=True)
                geometry.export(f"{cfg.output_path}/mesh{it:05d}", f"{it:05d}")

                if verbose:
                    chosen_idx = np.random.randint(0, batch["img"].shape[0])
                    opt_img = out["shaded"][chosen_idx].clone().detach()
                    # save images
                    img = opt_img.cpu().numpy()

                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_ours-{}.png".format(cfg.output_path, it))

                    img = color_ref[chosen_idx].cpu().numpy()
                    print(img.shape)
                    if img.shape[2] == 1:
                        img = np.concatenate([img, img, img, img], axis=2)

                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("{}/a_gt-{}.png".format(cfg.output_path, it))

                    if cfg.get("fitting_stage", None) == "geometry":
                        diff = color_ref[chosen_idx].cpu().numpy(
                        )[..., -1:] - opt_img.cpu().numpy()[..., -1:]
                        # save images
                        img = np.abs(diff)
                        print(img.shape)
                        if img.shape[2] == 1:
                            img = np.concatenate([img, img, img, img], axis=2)

                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save("{}/a_diff-{}.png".format(cfg.output_path, it))

    print(f"Best rendering loss: {best_loss} at iteration {best_loss_iter}")
    geometry.export(f"{cfg.output_path}/final", "final", save_npy=True)

    if material is not None:
        material.export(f"{cfg.output_path}/final", "material")
        renderer.export(f"{cfg.output_path}/final", "material")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    args, extras = parser.parse_known_args()
    
    cfg_file = args.config
    cfg = load_config(cfg_file, cli_args=extras)
    
    train(cfg)
