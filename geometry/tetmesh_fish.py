import pypgo
from dataclasses import dataclass

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

from .tetmesh_geometry import TetMeshMultiSphereGeometry, TetrahedronMesh  # NOQA

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import parse_structured, get_device  # NOQA
from utils.typing import *  # NOQA


class TetMeshFish(TetMeshMultiSphereGeometry):
    @dataclass
    class Config(TetMeshMultiSphereGeometry.Config):
        fish_skeleton_file_path: str

    def __init__(self, cfg: Config):

        torch.nn.Module.__init__(self)

        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        # key points
        with open(self.cfg.fish_skeleton_file_path, "r") as fin:
            skel = json.load(fin)
            skel_edges = np.asarray(skel["edges"])
            num_spheres = skel_edges.shape[0]

        # scale smoothness coeff by the number of spheres
        if self.cfg.use_smooth_barrier:
            self.cfg.smooth_barrier_param["smooth_eng_coeff"] /= num_spheres

        # initialize tet spheres
        if not self.cfg.load_precomputed_tetwild_mesh:
            # initialize spheres
            tetwild_exe = self.cfg.tetwild_exec
            os.makedirs(self.cfg.tetwild_cache_folder, exist_ok=True)

            async def do_subprocess(cmd: str):
                print(f"Run {cmd}")
                proc = await asyncio.create_subprocess_shell(cmd)
                await proc.wait()

            loop = asyncio.get_event_loop()
            tasks = []
            for sp_i in range(num_spheres):
                # print(f"processing sphere {sp_i}...")
                # sphere_v = np.copy(template_sphere.vertices).astype(np.float32)
                # sphere_t = np.copy(template_sphere.faces).astype(
                #     np.int32).flatten()

                # sphere_v = sphere_v * skel_r[sp_i] + skel_pts[sp_i]
                # sphere_v = sphere_v.flatten()

                # template_sphere_trimeshgeo = pypgo.create_trimeshgeo(
                #     sphere_v, sphere_t)

                # trimesh_remeshed = pypgo.mesh_isotropic_remeshing(
                #     template_sphere_trimeshgeo, final_edge_length, 5, 180)
                # tri_v = pypgo.trimeshgeo_get_vertices(trimesh_remeshed)
                # tri_t = pypgo.trimeshgeo_get_triangles(trimesh_remeshed)
                centers = skel_edges[sp_i]["centers"]
                radii = skel_edges[sp_i]["radii"]

                template_sphere_edge_surface = pypgo.create_tetsphere_edge_surface(
                    centers, radii, "/home/mhg/Projects/libpog/mesh_data/s.1.obj")
                tri_v = pypgo.trimeshgeo_get_vertices(
                    template_sphere_edge_surface).reshape(-1, 3)
                tri_t = pypgo.trimeshgeo_get_triangles(
                    template_sphere_edge_surface).reshape(-1, 3)

                trimesh.Trimesh(vertices=tri_v.reshape(-1, 3), faces=tri_t.reshape(-1, 3)
                                ).export(f"{self.cfg.tetwild_cache_folder}/temp{sp_i}.obj")

                cmd = f"{tetwild_exe} --input {self.cfg.tetwild_cache_folder}/temp{sp_i}.obj --output {self.cfg.tetwild_cache_folder}/temp{sp_i}.msh --targeted-num-v {tri_v.shape[0]} --epsilon 0.001 --is-quiet"
                tasks.append(asyncio.ensure_future(do_subprocess(cmd)))

            loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()

            all_v_param = []
            all_tet_faces = []

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
                base_vid += all_v_param[-1].shape[0]

            all_v_param = np.concatenate(all_v_param, axis=0)
            all_tet_faces = np.concatenate(all_tet_faces, axis=0)

            print(f"final v: {all_v_param.shape}")
            print(f"final t: {all_tet_faces.shape}")

            v = all_v_param
            f = all_tet_faces

            np.save(f"{self.cfg.tetwild_cache_folder}/final_tet_v.npy", v)
            np.save(f"{self.cfg.tetwild_cache_folder}/final_tet_t.npy", f)

        else:
            v = np.load(f"{self.cfg.tetwild_cache_folder}/final_tet_v.npy")
            f = np.load(f"{self.cfg.tetwild_cache_folder}/final_tet_t.npy")

        ###
        self.tetmesh = TetrahedronMesh(vtx_np=v, elem_np=f)
        if self.cfg.debug_mode:
            # dump meshes
            self.tetmesh.save("debug", "debug_multi_spheres",
                              save_surface_mesh=True)

        self.setup()
