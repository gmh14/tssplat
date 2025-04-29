import pypgo

import sys
import os
import json
import numpy as np
import glob
from PIL import Image
import mcubes
import trimesh
import struct
import time
import argparse

import torch
from scipy.optimize import milp, Bounds, LinearConstraint

class SignDistanceFunction:

    def __init__(self, dim, bmin, bmax, dev) -> None:
        self.dev = dev
        self.sdf = torch.zeros((dim, dim, dim)).to(dev)
        self.bmin = bmin.to(dev)
        self.bmax = bmax.to(dev)
        self.dx = ((bmax - bmin) / (dim - 1)).to(self.dev)
        self.dim = dim

    def get_pos(self, idx):
        return idx.to(self.dev).to(dtype=torch.float32, device=self.dev) * self.dx + self.bmin

    def save_vega(self, filename):
        with open(filename, "wb") as f:
            f.write(struct.pack("i", -self.dim))
            f.write(struct.pack("i", self.dim))
            f.write(struct.pack("i", self.dim))

            f.write(struct.pack("d", self.bmin[0].item()))
            f.write(struct.pack("d", self.bmin[1].item()))
            f.write(struct.pack("d", self.bmin[2].item()))

            f.write(struct.pack("d", self.bmax[0].item()))
            f.write(struct.pack("d", self.bmax[1].item()))
            f.write(struct.pack("d", self.bmax[2].item()))

            d = self.sdf.cpu().numpy().astype(np.float32)
            bytes = d.tobytes()

            f.write(bytes)


def load_data(tgt_path, rendering_type):
    all_tgt_imgs = []
    all_mvp_mats = []
    
    if rendering_type == "mistuba":
        # tgt_path = sys.argv[2]
        for img_file in glob.glob(os.path.join(tgt_path, "img*rgba*.png")):
            print(img_file)
            tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            all_tgt_imgs.append(tgt_img)

            img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]

            mvp_mat_file = os.path.join(tgt_path, "mvp_mtx_{}.npy".format(img_id))
            mvp_mat = np.load(mvp_mat_file)
            all_mvp_mats.append(mvp_mat.astype(np.float32))
            assert np.all(np.isfinite(mvp_mat))
    elif rendering_type == "1":
        images = np.load(sys.argv[2])
        mvps = np.load(sys.argv[3])

        for i in range(images.shape[0]):
            all_tgt_imgs.append(images[i])
            all_mvp_mats.append(mvps[i])

            assert np.all(np.isfinite(all_tgt_imgs[-1]))
            assert np.all(np.isfinite(all_mvp_mats[-1]))
    elif int(rendering_type) == 2:
        camera_mvp_root = "F:/work/Dropbox/doc2/skl-shape-gen/AllData/GSO/wonder3d_output/wonder3d_mvp"
        camera_files = [
            "back_left",
            "back_right",
            "front_left",
            "front_right",
            "front",
            "back",
            "left",
            "right",
        ]

        for view in camera_files:
            camera_filename = f"{camera_mvp_root}/{view}_mvp.npy"
            all_mvp_mats.append(np.load(camera_filename))
            all_tgt_imgs.append(np.zeros(1))

        img_root = sys.argv[2]
        for img_file in os.listdir(img_root):
            found = False
            for i, c in enumerate(camera_files):
                if c in img_file:
                    found = True
                    tgt_img = np.array(Image.open(os.path.join(img_root, img_file))).astype(np.float32) / 255.0
                    all_tgt_imgs[i] = tgt_img
                    break

            assert found

        imgs = []
        mvps = []
        for mvp, img in zip(all_mvp_mats, all_tgt_imgs):
            if len(img.shape) == 3:
                imgs.append(img)
                mvps.append(mvp.astype(np.float32))

        all_tgt_imgs = imgs
        all_mvp_mats = mvps
        
    return all_tgt_imgs, all_mvp_mats


def generate_rough_sdf(all_tgt_imgs, all_mvp_mats, dim, filename, gen_pcd):
    print(all_tgt_imgs[0].shape)
    w, h = all_tgt_imgs[0].shape[:2]
    print(f"image size:({w},{h})")

    assert w == h
    res = w

    device = "cuda:0"
    bmin = torch.ones(3) * -1.2
    bmax = torch.ones(3) * 1.2
    sdf = SignDistanceFunction(dim, bmin, bmax, device)

    # create indices
    indices = torch.tensor(list(range(dim))).to(device)
    grid_x, grid_y, grid_z = torch.meshgrid(indices, indices, indices, indexing="ij")
    idx_pos = torch.concat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)], dim=1)

    # create positions
    world_pos = sdf.get_pos(idx_pos)

    # print(world_pos[0:3, :])
    world_pos4 = torch.concat([world_pos, torch.ones_like(world_pos[:, 0:1])], dim=1)
    print(f"shape:{world_pos4.shape}")

    mvp_all = np.stack(all_mvp_mats, axis=0)
    mvp_all = torch.from_numpy(mvp_all).to(device)
    # print(mvp_all.shape)

    do_all = False
    if do_all:
        # (b, 4, 4) x (v, 4)
        mvp_all_for_mul = mvp_all.reshape(1, mvp_all.shape[0], 4, 4).expand(world_pos4.shape[0], -1, -1, -1)

        world_pos4_for_mul = world_pos4.reshape(world_pos4.shape[0], 1, 4, 1).expand(-1, mvp_all.shape[0], -1, -1)

        # (v, b, 4, 1)
        transformed_pos4 = torch.matmul(mvp_all_for_mul, world_pos4_for_mul).squeeze()
        transformed_pos4 /= transformed_pos4[:, :, 3:4]

        img_coord = (transformed_pos4[:, :, 0:2] * 0.5 + 0.5) * res

        mask = torch.ones(img_coord.shape[0]).bool().to(device)
        for img_i in range(len(all_tgt_imgs)):
            img = torch.from_numpy(all_tgt_imgs[img_i]).to(device)
            img = img[..., 3]
            img_coord_truncated = torch.clamp(img_coord[:, img_i, :].to(dtype=torch.int32), 0, res - 1)

            mask = torch.logical_and(mask, img[img_coord_truncated[:, 1], img_coord_truncated[:, 0]] > 0.5)
            mask = torch.logical_and(mask, img_coord[:, img_i, 0] >= 0)
            mask = torch.logical_and(mask, img_coord[:, img_i, 0] < res)
            mask = torch.logical_and(mask, img_coord[:, img_i, 1] >= 0)
            mask = torch.logical_and(mask, img_coord[:, img_i, 1] < res)

            print(img_i, end=" ")
            sys.stdout.flush()
    else:
        mask = torch.ones(world_pos.shape[0]).bool().to(device)

        # (4, 4) x (v, 4)
        for img_i in range(len(all_tgt_imgs)):
            mvp = mvp_all[img_i, ...]
            world_pos4_for_mul = world_pos4.reshape(world_pos4.shape[0], 4, 1)

            # (v, b, 4, 1)
            transformed_pos4 = torch.matmul(mvp, world_pos4_for_mul).squeeze()
            transformed_pos4 /= transformed_pos4[:, 3:4]
            img_coord = (transformed_pos4[:, 0:2] * 0.5 + 0.5) * res

            img = torch.from_numpy(all_tgt_imgs[img_i]).to(device)
            img = img[..., 3]
            img_coord_truncated = torch.clamp(img_coord[:, :].to(dtype=torch.int32), 0, res - 1)

            mask = torch.logical_and(mask, img[img_coord_truncated[:, 1], img_coord_truncated[:, 0]] > 0.01)

            print(img_i, end=" ")
            sys.stdout.flush()
            
    if gen_pcd == True:
        with open(filename, "w") as f:
            for i in range(mask.shape[0]):
                if mask[i]:
                    f.write(f"v {world_pos[i,0]} {world_pos[i,1]} {world_pos[i,2]}\n")
    else:
        sdf.sdf[idx_pos[:, 0], idx_pos[:, 1], idx_pos[:, 2]] = torch.where(
            mask,
            torch.ones(img_coord.shape[0]).to(device),
            torch.ones(img_coord.shape[0]).to(device) * -1,
        )

        binary_sdf = sdf.sdf.cpu().numpy() > 0
        smoothed_sdf = binary_sdf  # mcubes.smooth(binary_sdf)
        v, f = mcubes.marching_cubes(smoothed_sdf, 0.01)
        v *= sdf.dx[0].item()
        v += sdf.bmin.cpu().detach().numpy()

        trimesh.Trimesh(vertices=v, faces=f).export(filename)


def generate_spheres(input_mesh_filename, sample_mesh_filename, radius_scale, offset):
    input_mesh = trimesh.load_mesh(input_mesh_filename)
    sample_mesh = trimesh.load_mesh(sample_mesh_filename)

    point_set = torch.tensor(input_mesh.vertices).cuda().double()
    inner_set = torch.tensor(sample_mesh.vertices).cuda().double()
    dist = torch.cdist(inner_set, point_set, p=2)
    radius = dist.topk(1, largest=False).values
    radius = radius * radius_scale + offset
    
    # coverage matrix
    radius_g = radius.permute(1, 0).repeat(len(point_set), 1)
    D = torch.gt(radius_g, dist.permute(1, 0)).type(torch.int)
    D = D.cpu().numpy()
    
    c = np.ones(len(inner_set))
    options = {"disp": True, "time_limit": 1000}
    A,b =  D, np.ones(len(point_set))
    integrality = np.ones(len(inner_set))
    lb, ub = np.zeros(len(inner_set)), np.ones(len(inner_set))
    variable_bounds = Bounds(lb, ub)
    constraints = LinearConstraint(A, lb=b)
    res_milp = milp(
        c,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options)

    res_milp.x = [int(x_i) for x_i in res_milp.x]
    print(res_milp)
    print(np.sum(res_milp.x))
    value_pos = np.nonzero(res_milp.x)[0]
    
    final_selected_pts = inner_set.cpu().numpy()[value_pos]
    final_radius = radius.cpu().numpy()[value_pos]
    
    return final_selected_pts, final_radius
    
def main(tgt_path, mesh_name, save_path, rendering_type, radius_scale=1.1, offset=0.06, pc_res=50, surf_res=50):
    img, mvp = load_data(tgt_path, rendering_type)
    os.makedirs(save_path, exist_ok=True)

    t1 = time.time()
    
    surf_mesh_path = os.path.join(save_path, "{}_surf.obj".format(mesh_name))
    pt_mesh_path = os.path.join(save_path, "{}_pt.obj".format(mesh_name))
    
    generate_rough_sdf(img, mvp, pc_res,  pt_mesh_path, True)
    print("generate pc done")
    generate_rough_sdf(img, mvp, surf_res, surf_mesh_path, False)
    print("generate surf done")
    
    select_pts, radius = generate_spheres(surf_mesh_path, pt_mesh_path, radius_scale, offset)

    t2 = time.time()
    print(f"time cost: {t2 - t1}")
    
    save_dict = {
        "pt": select_pts.tolist(),
        "r": radius.tolist()
    }
    
    # save to json
    with open(os.path.join(save_path, "{}.json".format(mesh_name)), "w") as f:
        json.dump(save_dict, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="../img_data/a_white_dog", help="path to mv images")
    parser.add_argument("--expr_name", default="a_white_dog",help="name of the expr")
    parser.add_argument("--save_path", default="../mesh_data/a_white_dog",help="path for saving")
    parser.add_argument("--rendering_type", default="mistuba", choices=["mistuba", "wonder3d"], help="type of rendering")
    
    parser.add_argument("--radius_scale", default=1.1, type=float, help="scale of radius")
    parser.add_argument("--offset", default=0.06, type=float, help="offset of radius")
    parser.add_argument("--surf_res", default=50, type=int, help="the grid resolution of coarse voxel grid")
    parser.add_argument("--pc_res", default=50, type=int, help="the grid resolution controlling the number of initial candicate points ")
    args = parser.parse_args()

    main(args.img_path, args.expr_name, args.save_path, args.rendering_type,
        radius_scale=args.radius_scale, offset=args.offset, pc_res=args.pc_res, surf_res=args.surf_res)
    