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
import tqdm

import torch
from scipy.optimize import milp, Bounds, LinearConstraint, linprog
from torch_max_mem import maximize_memory_utilization

sys.path.append(os.path.dirname(__file__))
from utils import get_full_min_sdf_skeleton, get_init_skel, get_min_sdf_skel


@maximize_memory_utilization()
def batch_construct_D(radius, dist, batch_size):
    return torch.cat(
        [
            torch.gt(radius, dist[start: start + batch_size]).type(torch.int)
            for start in range(0, dist.shape[0], batch_size)
        ],
        dim=0,
    )


@maximize_memory_utilization()
def batch_cdist(x, y, batch_size):
    return torch.cat(
        [
            torch.cdist(x[start: start + batch_size], y)
            for start in range(0, x.shape[0], batch_size)
        ],
        dim=0,
    )


def get_surface_vf(faces):
    # get surface faces
    org_triangles = np.vstack(
        [
            faces[:, [1, 2, 3]],
            faces[:, [0, 3, 2]],
            faces[:, [0, 1, 3]],
            faces[:, [0, 2, 1]],
        ]
    )

    # Sort each triangle's vertices to avoid duplicates due to ordering
    triangles = np.sort(org_triangles, axis=1)

    unique_triangles, tri_idx, counts = np.unique(
        triangles, axis=0, return_index=True, return_counts=True
    )

    once_tri_id = counts == 1
    surface_triangles = unique_triangles[once_tri_id]

    surface_vertices = np.unique(surface_triangles)

    vertex_mapping = {vertex_id: i for i,
                      vertex_id in enumerate(surface_vertices)}

    mapped_triangles = np.vectorize(vertex_mapping.get)(
        org_triangles[tri_idx][once_tri_id]
    )

    return surface_vertices, mapped_triangles


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


def load_data(tgt_path):
    all_tgt_imgs = []
    all_mvp_mats = []

    for img_file in tqdm.tqdm(glob.glob(os.path.join(tgt_path, "img*rgba*.png"))):
        tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
        all_tgt_imgs.append(tgt_img)

        img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]

        mvp_mat_file = os.path.join(
            tgt_path, "mvp_mtx_{}.npy".format(img_id))
        mvp_mat = np.load(mvp_mat_file)
        all_mvp_mats.append(mvp_mat.astype(np.float32))
        assert np.all(np.isfinite(mvp_mat))
        
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
    grid_x, grid_y, grid_z = torch.meshgrid(
        indices, indices, indices, indexing="ij")
    idx_pos = torch.concat(
        [grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)], dim=1)

    # create positions
    world_pos = sdf.get_pos(idx_pos)

    # print(world_pos[0:3, :])
    world_pos4 = torch.concat(
        [world_pos, torch.ones_like(world_pos[:, 0:1])], dim=1)
    print(f"shape:{world_pos4.shape}")

    mvp_all = np.stack(all_mvp_mats, axis=0)
    mvp_all = torch.from_numpy(mvp_all).to(device)
    # print(mvp_all.shape)

    do_all = False
    if do_all:
        # (b, 4, 4) x (v, 4)
        mvp_all_for_mul = mvp_all.reshape(1, mvp_all.shape[0], 4, 4).expand(
            world_pos4.shape[0], -1, -1, -1)

        world_pos4_for_mul = world_pos4.reshape(
            world_pos4.shape[0], 1, 4, 1).expand(-1, mvp_all.shape[0], -1, -1)

        # (v, b, 4, 1)
        transformed_pos4 = torch.matmul(
            mvp_all_for_mul, world_pos4_for_mul).squeeze()
        transformed_pos4 /= transformed_pos4[:, :, 3:4]

        img_coord = (transformed_pos4[:, :, 0:2] * 0.5 + 0.5) * res

        mask = torch.ones(img_coord.shape[0]).bool().to(device)
        for img_i in range(len(all_tgt_imgs)):
            img = torch.from_numpy(all_tgt_imgs[img_i]).to(device)
            img = img[..., 3]
            img_coord_truncated = torch.clamp(
                img_coord[:, img_i, :].to(dtype=torch.int32), 0, res - 1)

            mask = torch.logical_and(
                mask, img[img_coord_truncated[:, 1], img_coord_truncated[:, 0]] > 0.5)
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
            img_coord_truncated = torch.clamp(
                img_coord[:, :].to(dtype=torch.int32), 0, res - 1)

            mask = torch.logical_and(
                mask, img[img_coord_truncated[:, 1], img_coord_truncated[:, 0]] > 0.01)

            print(img_i, end=" ")
            sys.stdout.flush()

    if gen_pcd == True:
        with open(filename, "w") as f:
            for i in range(mask.shape[0]):
                if mask[i]:
                    f.write(
                        f"v {world_pos[i,0]} {world_pos[i,1]} {world_pos[i,2]}\n")
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


def generate_spheres_old(input_mesh_filename, sample_mesh_filename, radius_scale, offset, save_path):
    input_mesh = trimesh.load_mesh(input_mesh_filename)
    sample_mesh = trimesh.load_mesh(sample_mesh_filename)

    point_set = torch.tensor(input_mesh.vertices).cuda().double()
    raw_inner_set = torch.tensor(sample_mesh.vertices).cuda().double()
    dist = batch_cdist(raw_inner_set, point_set, batch_size=500)
    
    # filter out the inner points that are too far from the surface
    min_dist_per_point = torch.min(dist, dim=1).values
    
    min_dist = torch.min(min_dist_per_point)
    max_dist = torch.max(min_dist_per_point)
    print(f"min_dist:{min_dist.cpu().numpy()}")
    print(f"max_dist:{max_dist.cpu().numpy()}")
    ratio = 0.3
    threshold = (1 - ratio) * min_dist + ratio * max_dist
    
    inner_set = raw_inner_set[min_dist_per_point < threshold]

    print(f"raw_inner_set:{raw_inner_set.shape}")
    print(f"inner_set:{inner_set.shape}")
    inner_set_mesh = trimesh.Trimesh(vertices=inner_set.cpu().numpy(), faces=[])
    inner_set_mesh.export(os.path.join(save_path, "inner_set_filtered.obj"))
    #
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    radius = dist.topk(10, largest=False).values.mean(dim=1, keepdim=True)
    radius = radius * radius_scale + offset

    dist = dist.permute(1, 0)  # [N, Nin]
    radius_g = radius.permute(1, 0)  # [1, Nin]

    D = batch_construct_D(radius_g, dist, batch_size=500)
    D = D.cpu().numpy()

    c = np.ones(len(inner_set))
    options = {"disp": True, "time_limit": 300}
    A, b = D, np.ones(len(point_set))
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
    binary_x = res_milp.x
    
    # res_lp = linprog(
    #     c,
    #     A_ub=D * -1,
    #     b_ub=b * -1,
    #     bounds=[(0, 1) for _ in range(len(inner_set))],
    #     options=options
    # )
    # threshold = 0.3  # Adjust threshold based on your problem's requirements
    # binary_x = (res_lp.x > threshold).astype(int)
    # print(res_lp)
    print(np.sum(binary_x))
    value_pos = np.nonzero(binary_x)[0]

    final_selected_pts = inner_set.cpu().numpy()[value_pos]
    final_radius = radius.cpu().numpy()[value_pos]

    return final_selected_pts, final_radius


def generate_spheres_v1(input_mesh_filename, sample_mesh_filename, radius_scale, offset, save_path):
    input_mesh = trimesh.load_mesh(input_mesh_filename)
    sample_mesh = trimesh.load_mesh(sample_mesh_filename)
    input_mesh_vtx_sampled = input_mesh.vertices

    input_mesh_vtx_sampled = input_mesh.sample(50000)
    input_mesh = trimesh.Trimesh(vertices=input_mesh_vtx_sampled, faces=[])
    input_mesh.export(os.path.join(save_path, "input_mesh_sampled.obj"))
    
    point_set = torch.tensor(input_mesh_vtx_sampled).cuda().double()
    raw_inner_set = torch.tensor(sample_mesh.vertices).cuda().double()
    dist = batch_cdist(raw_inner_set, point_set, batch_size=500)
    
    # filter out the inner points that are too far from the surface
    min_dist_per_point = torch.min(dist, dim=1).values
    
    min_dist = torch.min(min_dist_per_point)
    max_dist = torch.max(min_dist_per_point)
    print(f"min_dist:{min_dist.cpu().numpy()}")
    print(f"max_dist:{max_dist.cpu().numpy()}")
    ratio = 0.3
    threshold = (1 - ratio) * min_dist + ratio * max_dist
    
    inner_set = raw_inner_set[min_dist_per_point < threshold]

    print(f"raw_inner_set:{raw_inner_set.shape}")
    print(f"inner_set:{inner_set.shape}")
    inner_set_mesh = trimesh.Trimesh(vertices=inner_set.cpu().numpy(), faces=[])
    inner_set_mesh.export(os.path.join(save_path, "inner_set_filtered.obj"))
    #
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    radius = dist.topk(10, largest=False).values.mean(dim=1, keepdim=True)
    radius = radius * radius_scale + offset

    dist = dist.permute(1, 0)  # [N, Nin]
    radius_g = radius.permute(1, 0)  # [1, Nin]

    D = batch_construct_D(radius_g, dist, batch_size=500)
    D = D.cpu().numpy()

    c = np.ones(len(inner_set))
    options = {"disp": True, "time_limit": 300}
    A, b = D, np.ones(len(point_set))
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
    binary_x = res_milp.x
    
    # res_lp = linprog(
    #     c,
    #     A_ub=D * -1,
    #     b_ub=b * -1,
    #     bounds=[(0, 1) for _ in range(len(inner_set))],
    #     options=options
    # )
    # threshold = 0.3  # Adjust threshold based on your problem's requirements
    # binary_x = (res_lp.x > threshold).astype(int)
    # print(res_lp)
    print(np.sum(binary_x))
    value_pos = np.nonzero(binary_x)[0]

    final_selected_pts = inner_set.cpu().numpy()[value_pos]
    final_radius = radius.cpu().numpy()[value_pos]

    return final_selected_pts, final_radius


def solve_milp(inner_set, point_set, radius, radius_scale, offset, options):
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    # radius = torch.tensor((candidate_points - input_mesh_remeshed.vertices).norm(dim=1, keepdim=True)).cuda().double()
    radius_scaled = radius * radius_scale + offset

    dist = dist.permute(1, 0)  # [N, Nin]
    radius_g = radius_scaled.permute(1, 0)  # [1, Nin]

    D = batch_construct_D(radius_g, dist, batch_size=500)
    D = D.cpu().numpy()
    
    # filter
    zero_rows = np.all(D == 0, axis=1)
    zero_row_count = np.sum(zero_rows)
    print(f"zero_row_count:{zero_row_count}")
    if zero_row_count < 200:
        D = D[~zero_rows]
        point_set = point_set[~zero_rows]

    c = np.ones(len(inner_set))
    A, b = D, np.ones(len(point_set))
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
    
    return res_milp, D, point_set


def generate_spheres(input_mesh_filename, sample_mesh_filename, radius_scale, offset, remesh_edge_length, save_path):
    input_mesh = trimesh.load_mesh(input_mesh_filename)
    
    # remesh
    trimeshgeo = pypgo.create_trimeshgeo(
        input_mesh.vertices.flatten(), input_mesh.faces.flatten())

    trimesh_remeshed = pypgo.mesh_isotropic_remeshing(
        trimeshgeo, remesh_edge_length, 5, 180)
    tri_v = pypgo.trimeshgeo_get_vertices(trimesh_remeshed)
    tri_t = pypgo.trimeshgeo_get_triangles(trimesh_remeshed)
    input_mesh_remeshed = trimesh.Trimesh(vertices=tri_v.reshape(-1, 3), faces=tri_t.reshape(-1, 3))
    input_mesh_remeshed.export(os.path.join(save_path, "input_mesh_remeshed.obj"))
    # sample_mesh = trimesh.load_mesh(sample_mesh_filename)
    
    skeleton = get_min_sdf_skel(input_mesh_remeshed)
    candidate_points = skeleton
    print(f"candidate_points:{candidate_points.shape}")
    
    point_set = torch.tensor(input_mesh.vertices).cuda().double()
    inner_set = torch.tensor(candidate_points).cuda().double()
    
    inner_set_mesh = trimesh.Trimesh(vertices=inner_set.cpu().numpy(), faces=[])
    inner_set_mesh.export(os.path.join(save_path, "inner_set_filtered.obj"))
    
    
    #
    options = {"disp": True, "time_limit": 30000, "mip_rel_gap": 0.20}
    dist = batch_cdist(inner_set, point_set, batch_size=500)
    radius = dist.topk(10, largest=False).values.mean(dim=1, keepdim=True)
    
    ## debug
    template_sphere = trimesh.creation.uv_sphere(radius=1.0)
    debug_vtx = []
    debug_faces = []
    vtx_offset = 0
    for ct, r in zip(inner_set.cpu().numpy().tolist(), (radius.cpu().numpy() * radius_scale + offset).tolist()):
        scaled_vertices = template_sphere.vertices.copy() * r + ct
        debug_vtx.append(scaled_vertices)
        
        offset_faces = template_sphere.faces.copy() + vtx_offset
        debug_faces.append(offset_faces)
        
        vtx_offset += len(template_sphere.vertices)

    # Combine all vertices and faces into a single mesh
    debug_mesh = trimesh.Trimesh(
        vertices=np.vstack(debug_vtx),
        faces=np.vstack(debug_faces))
    debug_mesh.export(os.path.join(save_path, "debug_mesh.obj"))
        
    
    res_milp, D, point_set = solve_milp(inner_set, point_set, radius, radius_scale, offset, options)
    
    res_milp.x = [int(x_i) for x_i in res_milp.x]
    binary_x = res_milp.x
    value_pos = np.nonzero(binary_x)[0]
    print("phase 1: ", np.sum(binary_x))
    
    ## check if there are points that are not covered
    covered_flag = D @ binary_x
    uncovered_flag = covered_flag < 0.5
    uncovered_points = point_set[uncovered_flag]
    print(f"uncovered_points:{np.sum(uncovered_flag)}")
    
    if np.sum(uncovered_flag) > 0:
        # solve again
        options = {"disp": True, "time_limit": 30000, "mip_rel_gap": 0.0}
        res_milp_2, _,  _ = solve_milp(inner_set, uncovered_points, radius, radius_scale, offset, options)
        res_milp_2.x = [int(x_i) for x_i in res_milp_2.x]
        binary_x_2 = res_milp_2.x
        print("phase 2: ", np.sum(binary_x_2))
        value_pos_2 = np.nonzero(binary_x_2)[0]
    
        value_pos = np.concatenate([value_pos, value_pos_2])
    
    print("all: ", value_pos.shape)

    final_selected_pts = inner_set.cpu().numpy()[value_pos]
    radius_scaled = radius * radius_scale + offset
    radius_scaled += offset * 0.3
    final_radius = radius_scaled.cpu().numpy()[value_pos]

    return final_selected_pts, final_radius


def main(tgt_path, mesh_name, save_path, radius_scale=1.1, offset=0.06, pc_res=50, surf_res=50, remesh_edge_length=0.08):
    img, mvp = load_data(tgt_path)
    os.makedirs(save_path, exist_ok=True)

    t1 = time.time()
    surf_mesh_path = os.path.join(save_path, "{}_surf.obj".format(mesh_name))
    pt_mesh_path = os.path.join(save_path, "{}_pt.obj".format(mesh_name))

    ################

    # generate_rough_sdf(img, mvp, pc_res,  pt_mesh_path, True)
    # print("generate pc done")
    generate_rough_sdf(img, mvp, surf_res, surf_mesh_path, False)
    print("generate surf done")
    ################

    select_pts, radius = generate_spheres(
        surf_mesh_path, pt_mesh_path, radius_scale, offset, remesh_edge_length, save_path)
    # surf_mesh = trimesh.load_mesh(surf_mesh_path)
    # pts, _ = get_full_min_sdf_skeleton(surf_mesh)
    # select_pts = pts.cpu().numpy()
    # trimesh.Trimesh(vertices=select_pts, faces=[]).export(os.path.join(
    #     save_path, "{}_skeleton.obj".format(mesh_name)))
    
    t2 = time.time()
    print(f"time cost: {t2 - t1}")

    save_dict = {
        "pt": select_pts.tolist(),
        "r": radius.tolist()
    }

    # save to json
    with open(os.path.join(save_path, "{}.json".format(mesh_name)), "w") as f:
        json.dump(save_dict, f, indent=4)

    final_selected_pc = trimesh.Trimesh(vertices=select_pts, faces=[])
    final_selected_pc.export(os.path.join(
        save_path, "{}_final_pc.obj".format(mesh_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", default="../img_data/a_white_dog", help="path to mv images")
    parser.add_argument("--expr_name", default="a_white_dog",
                        help="name of the expr")
    parser.add_argument(
        "--save_path", default="../mesh_data/a_white_dog", help="path for saving")

    parser.add_argument("--radius_scale", default=1.1,
                        type=float, help="scale of radius")
    parser.add_argument("--offset", default=0.06,
                        type=float, help="offset of radius")
    parser.add_argument("--surf_res", default=50, type=int,
                        help="the grid resolution of coarse voxel grid")
    parser.add_argument("--pc_res", default=50, type=int,
                        help="the grid resolution controlling the number of initial candicate points")
    parser.add_argument("--remesh_edge_length", default=0.08,
                        type=float)
    args = parser.parse_args()

    main(args.img_path, args.expr_name, args.save_path,
         radius_scale=args.radius_scale, offset=args.offset, pc_res=args.pc_res, surf_res=args.surf_res, remesh_edge_length=args.remesh_edge_length)
