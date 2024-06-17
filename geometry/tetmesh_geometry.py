from dataclasses import dataclass, field
import pypgo

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
import json
import math
import trimesh
import asyncio

from .tetrahedron_mesh import TetrahedronMesh

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import parse_structured, get_device  # NOQA
from utils.typing import *  # NOQA
from energies.smooth_barrier import SmoothnessBarrierEnergy  # NOQA


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


class TetMeshGeometryForwardData:
    def __init__(self, tet_v: torch.Tensor, tet_elem: torch.Tensor, surface_vid: torch.Tensor, surface_f: torch.Tensor, smooth_barrier_energy: Optional[SmoothnessBarrierEnergy] = None):
        self.tet_v = tet_v
        self.tet_elem = tet_elem

        # surface
        self.v_pos = tet_v[surface_vid]
        self.t_pos_idx = surface_f

        # geometry regularization
        self.smooth_barrier_energy = smooth_barrier_energy

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm,
                v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - \
            uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(
                denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            # tangents[n_i] = tangents[n_i] + tang
            tangents.scatter_add_(0, idx, tang)
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(
            tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents


class TetMeshGeometry(torch.nn.Module):
    @dataclass
    class Config:
        use_smooth_barrier: bool
        initial_mesh_path: Optional[str]
        smooth_barrier_param: Optional[dict]
        optimize_geo: bool

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        super(TetMeshGeometry, self).__init__()

        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.tetmesh = TetrahedronMesh(veg_file_path=cfg.initial_mesh_path)
        self.setup()

    def setup(self):
        tet_v = torch.from_numpy(self.tetmesh.vtx_init).to(
            torch.float32).to(self.device)
        if self.cfg.optimize_geo:
            self.tet_v = torch.nn.Parameter(tet_v, requires_grad=True)
        else:
            self.register_buffer("tet_v", tet_v)

        self.tet_elem = torch.from_numpy(
            self.tetmesh.elem).to(torch.int32).to(self.device)
        self.surface_vid = torch.from_numpy(
            self.tetmesh.surface_vid).to(torch.int32).to(self.device)
        self.surface_fid = torch.from_numpy(
            self.tetmesh.surface_fid).to(torch.int32).to(self.device)

        self.uv_idx = torch.from_numpy(self.tetmesh.uv_idx.astype(
            np.int32)).to(torch.int32).to(self.device)
        self.uv = torch.from_numpy(self.tetmesh.uv).to(
            torch.float32).to(self.device)

        self.mesh_smooth_barrier = None
        if self.cfg.use_smooth_barrier:
            assert "smooth_eng_coeff" in self.cfg.smooth_barrier_param
            assert "barrier_coeff" in self.cfg.smooth_barrier_param
            assert "increase_order_iter" in self.cfg.smooth_barrier_param

            self.mesh_smooth_barrier = SmoothnessBarrierEnergy(
                self.tetmesh.vtx_init, self.tetmesh.elem, self.cfg.smooth_barrier_param)

    def reset(self, tet_v_np: np.array, tet_elem_np: np.array, surface_vid_np: Optional[np.array] = None, surface_fid_np: Optional[np.array] = None):
        del self.tetmesh, self.tet_v, self.tet_elem, self.surface_vid, self.surface_fid, self.uv_idx, self.uv
        torch.cuda.empty_cache()

        self.tetmesh = TetrahedronMesh(
            vtx_np=tet_v_np, elem_np=tet_elem_np, surface_vid_np=surface_vid_np, surface_fid_np=surface_fid_np)
        self.setup()

    def remesh(self, cache_name="cache"):
        pass  # use tetwild to remesh

    def forward(self, iter_num, **kwargs):
        if "permute_surface_v" in kwargs:
            assert "permute_surface_v_dev" in kwargs
            dev = kwargs["permute_surface_v_dev"]
            print(f"permute surface v with dev {dev}")
            with torch.no_grad():
                self.tet_v[self.surface_vid] += torch.rand(
                    self.tet_v[self.surface_vid].shape, device=self.device) * dev - dev * 0.5

        smooth_barrier_energy = None
        if self.cfg.use_smooth_barrier:
            smooth_coeff, barrier_coeff = self.mesh_smooth_barrier.coeff_scheduler(
                iter_num)
            smooth_barrier_energy = self.mesh_smooth_barrier(
                self.tet_v, iter_num, smooth_coeff, barrier_coeff)

        forward_data = TetMeshGeometryForwardData(
            self.tet_v, self.tet_elem, self.surface_vid, self.surface_fid, smooth_barrier_energy)
        return forward_data

    def export(self, path: str, filename: str, **kwargs):
        self.tet_v_np = self.tet_v.clone().detach().cpu().numpy()
        self.tetmesh.update_vtx_pos(self.tet_v_np)
        self.tet_v_np = self.tet_v_np.reshape(-1, 3)
        self.tetmesh.save(path, filename, **kwargs)


class TetMeshMultiSphereGeometry(TetMeshGeometry):
    @dataclass
    class Config(TetMeshGeometry.Config):
        template_surface_sphere_path: str
        key_points_file_path: str

        tetwild_exec: str
        tetwild_cache_folder: str
        load_precomputed_tetwild_mesh: bool
        output_path: str

        debug_mode: bool

    def __init__(self, cfg: Config):

        torch.nn.Module.__init__(self)
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        if self.cfg.initial_mesh_path != "":
            self.tetmesh = TetrahedronMesh(veg_file_path=os.path.join(
                self.cfg.initial_mesh_path, "final.veg"))
            with open(f"{self.cfg.initial_mesh_path}/spheres_vtx_idx.json", "r") as fin:
                all_spheres_vtx_idx = json.load(fin)
            with open(f"{self.cfg.initial_mesh_path}/spheres_elem_idx.json", "r") as fin:
                all_spheres_elem_idx = json.load(fin)

            ###
            self.all_spheres_vtx_idx = all_spheres_vtx_idx
            self.all_spheres_elem_idx = all_spheres_elem_idx

        else:
            # key points
            with open(self.cfg.key_points_file_path, "r") as fin:
                skel = json.load(fin)
                skel_pts = np.asarray(skel["pt"])
                skel_r = np.asarray(skel["r"])
                num_spheres = skel_pts.shape[0]

            # scale smoothness coeff by the number of spheres
            if self.cfg.use_smooth_barrier:
                self.cfg.smooth_barrier_param["smooth_eng_coeff"] /= num_spheres

            # initialize tet spheres
            if not self.cfg.load_precomputed_tetwild_mesh:
                # initialize spheres
                tetwild_exe = self.cfg.tetwild_exec
                os.makedirs(self.cfg.tetwild_cache_folder, exist_ok=True)

                # compute min radius
                min_radius = min(skel_r)
                min_n_triangles = 100

                min_surface_area = min_radius * min_radius * math.pi
                min_triangle_area = min_surface_area / min_n_triangles

                # x * x * sqrt(3) / 2 / 2
                edge_length_wrt_triangle_count = math.sqrt(
                    min_triangle_area * 4.0 / math.sqrt(3))
                edge_length_wrt_bb = 0.03
                edge_length_min = 0.015

                final_edge_length = max(edge_length_min, min(
                    edge_length_wrt_bb, edge_length_wrt_triangle_count))
                print(f"edge length: {final_edge_length}")

                template_sphere = trimesh.load(
                    self.cfg.template_surface_sphere_path)

                async def do_subprocess(cmd: str):
                    print(f"Run {cmd}")
                    proc = await asyncio.create_subprocess_shell(cmd)
                    await proc.wait()

                loop = asyncio.get_event_loop()
                tasks = []
                for sp_i in range(num_spheres):
                    print(f"processing sphere {sp_i}...")
                    sphere_v = np.copy(
                        template_sphere.vertices).astype(np.float32)
                    sphere_t = np.copy(template_sphere.faces).astype(
                        np.int32).flatten()

                    sphere_v = sphere_v * skel_r[sp_i] + skel_pts[sp_i]
                    sphere_v = sphere_v.flatten()

                    template_sphere_trimeshgeo = pypgo.create_trimeshgeo(
                        sphere_v, sphere_t)

                    trimesh_remeshed = pypgo.mesh_isotropic_remeshing(
                        template_sphere_trimeshgeo, final_edge_length, 5, 180)
                    tri_v = pypgo.trimeshgeo_get_vertices(trimesh_remeshed)
                    tri_t = pypgo.trimeshgeo_get_triangles(trimesh_remeshed)

                    trimesh.Trimesh(vertices=tri_v.reshape(-1, 3), faces=tri_t.reshape(-1, 3)
                                    ).export(f"{self.cfg.tetwild_cache_folder}/temp{sp_i}.obj")

                    cmd = f"{tetwild_exe} --input {self.cfg.tetwild_cache_folder}/temp{sp_i}.obj --output {self.cfg.tetwild_cache_folder}/temp{sp_i}.msh --targeted-num-v {tri_v.shape[0]} --epsilon 0.001 --is-quiet"
                    tasks.append(asyncio.ensure_future(do_subprocess(cmd)))

                loop.run_until_complete(asyncio.gather(*tasks))
                loop.close()

                all_v_param = []
                all_tet_faces = []
                all_spheres_vtx_idx = []
                all_spheres_elem_idx = []

                base_vid = 0
                for sp_i in range(num_spheres):
                    tet_v = np.load(
                        f"{self.cfg.tetwild_cache_folder}/temp{sp_i}.msh_VO.npy")
                    tet_t = np.load(
                        f"{self.cfg.tetwild_cache_folder}/temp{sp_i}.msh_TO.npy")

                    print(f"tet v: {tet_v.shape}")
                    print(f"tet t: {tet_t.shape}")

                    all_v_param.append(tet_v.astype(np.float32))
                    all_tet_faces.append(tet_t.astype(np.int32) + base_vid)

                    all_spheres_vtx_idx.append(
                        list(range(base_vid, base_vid + tet_v.shape[0])))
                    all_spheres_elem_idx.append(
                        tet_t.astype(np.int32).tolist())

                    base_vid += all_v_param[-1].shape[0]

                all_v_param = np.concatenate(all_v_param, axis=0)
                all_tet_faces = np.concatenate(all_tet_faces, axis=0)

                print(f"final v: {all_v_param.shape}")
                print(f"final t: {all_tet_faces.shape}")

                v = all_v_param
                f = all_tet_faces

                np.save(f"{self.cfg.tetwild_cache_folder}/final_tet_v.npy", v)
                np.save(f"{self.cfg.tetwild_cache_folder}/final_tet_t.npy", f)

                with open(f"{self.cfg.output_path}/final/spheres_vtx_idx.json", "w") as fout:
                    json.dump(all_spheres_vtx_idx, fout, indent=4)
                with open(f"{self.cfg.output_path}/final/spheres_elem_idx.json", "w") as fout:
                    json.dump(all_spheres_elem_idx, fout, indent=4)

            else:
                v = np.load(f"{self.cfg.tetwild_cache_folder}/final_tet_v.npy")
                f = np.load(f"{self.cfg.tetwild_cache_folder}/final_tet_t.npy")

                with open(f"{self.cfg.output_path}/final/spheres_vtx_idx.json", "r") as fin:
                    all_spheres_vtx_idx = json.load(fin)
                with open(f"{self.cfg.output_path}/final/spheres_elem_idx.json", "r") as fin:
                    all_spheres_elem_idx = json.load(fin)

            ###
            self.all_spheres_vtx_idx = all_spheres_vtx_idx
            self.all_spheres_elem_idx = all_spheres_elem_idx

            self.tetmesh = TetrahedronMesh(vtx_np=v, elem_np=f)

        self.setup()

        if self.cfg.debug_mode:
            # dump meshes
            self.tetmesh.save("debug", "debug_multi_spheres",
                              save_surface_mesh=True)

    def permute_surface_v(self):
        self.tetmesh.permute_surface_v()
        self.setup()

    def export(self, path: str, filename: str, **kwargs):
        super().export(path, filename, **kwargs)
        # save tet mesh for each individual sphere
        for i in range(len(self.all_spheres_vtx_idx)):
            sphere_vtx = self.tet_v_np[self.all_spheres_vtx_idx[i], :]
            sphere_elem = self.all_spheres_elem_idx[i]
            np.save(os.path.join(path, filename +
                    "_sp{}_vtx.npy".format(i)), sphere_vtx)
            np.save(os.path.join(path, filename +
                    "_sp{}_elem.npy".format(i)), sphere_elem)
