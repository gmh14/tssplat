from dataclasses import dataclass, field
import nvdiffrast.torch as dr
import torch
import numpy as np
import cv2
import trimesh
import pymeshlab

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.typing import *  # NOQA
from utils.config import parse_structured, get_device  # NOQA
from utils.save import _save_mtl, _save_obj  # NOQA
from geometry.tetmesh_geometry import TetMeshGeometry  # NOQA
from materials import ExplicitMaterial  # NOQA


class MeshRasterizer(torch.nn.Module):
    @dataclass
    class Config:
        context_type: str
        is_orhto: bool

    def __init__(self, geometry: TetMeshGeometry,
                 materials: Optional[ExplicitMaterial],
                 cfg: Optional[Union[dict, DictConfig]]):
        super().__init__()

        self.cfg = parse_structured(self.Config, cfg)

        if self.cfg.context_type == "cuda":
            self.glctx = dr.RasterizeCudaContext()
        elif self.cfg.context_type == "gl":
            self.glctx = dr.RasterizeGLContext()
        else:
            raise ValueError("Invalid context type")

        self.device = get_device()

        self.geometry = geometry

        if materials is not None:
            self.materials = materials
        else:
            self.materials = None

        # dummy tensors
        self.ones_surface_v = torch.ones(
            [self.geometry.surface_vid.shape[0], 1]).to(self.device)
        self.zeros_surface_v = torch.zeros(
            [self.geometry.surface_vid.shape[0], 1]).to(self.device)

        self.tri_hash = None

    def transform_pos(self, mtx, pos, is_vec=False):
        t_mtx = torch.from_numpy(mtx).to(self.device) if isinstance(
            mtx, np.ndarray) else mtx.to(self.device)

        if is_vec:
            # posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
            posw = torch.cat([pos, self.zeros_surface_v],
                             axis=1).to(self.device)
            # return torch.matmul(posw, t_mtx.t())[None, ...]
            res = torch.matmul(posw, t_mtx.transpose(1, 2))  # [None, ...]
        else:
            # posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
            posw = torch.cat([pos, self.ones_surface_v],
                             axis=1).to(self.device)

            # return torch.matmul(posw, t_mtx.t())[None, ...]
            res = torch.matmul(posw, t_mtx.transpose(1, 2))  # [None, ...]

            # TODO check
            if self.cfg.is_orhto:
                res[..., 2] /= 6

        return res

    def forward(self, mvp: torch.Tensor,
                only_alpha: bool,
                iter_num: int, resolution: int,
                permute_surface_scheduler=None,
                fit_normal: bool = False, fit_depth: bool = False,
                background: Optional[torch.Tensor] = None,
                campos: Optional[torch.Tensor] = None):

        geo_input = {"iter_num": iter_num}

        if permute_surface_scheduler is not None:
            permute_dev = permute_surface_scheduler(iter_num)
            if permute_dev is not None:
                geo_input["permute_surface_v"] = True
                geo_input["permute_surface_v_dev"] = permute_dev

        geometry_forward_data = self.geometry(**geo_input)

        res = [resolution, resolution]

        pos_clip = self.transform_pos(
            mvp, geometry_forward_data.v_pos)  # , is_vec=True)
        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, geometry_forward_data.t_pos_idx, resolution=res, grad_db=False)

        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        alpha = dr.antialias(alpha.contiguous(), rast_out, pos_clip,
                             geometry_forward_data.t_pos_idx, topology_hash=self.tri_hash, pos_gradient_boost=1.0)

        shaded = alpha
        if not only_alpha:
            assert self.materials is not None
            assert background is not None
            mask = rast_out[..., -1:] > 0
            selector = mask[..., 0]

            positions_all, _ = dr.interpolate(
                geometry_forward_data.v_pos[None, ...], rast_out, geometry_forward_data.t_pos_idx)
            positions = positions_all[selector]

            color = self.materials(positions=positions)["color"]

            batch_size = rast_out[..., -1:].shape[0]
            gb_fg = torch.zeros(batch_size, res[0], res[0], 3).to(self.device)
            gb_fg[selector] = color
            gb_mat = torch.lerp(background, gb_fg, mask.float())

            gb_mat_aa = dr.antialias(gb_mat, rast_out, pos_clip,
                                     geometry_forward_data.t_pos_idx, topology_hash=self.tri_hash, pos_gradient_boost=1.0)

            # color = color * alpha + background * (1 - alpha)
            shaded = gb_mat_aa  # torch.cat([color, alpha], dim=-1)

        out = {"shaded": shaded,
               "geo_regularization": geometry_forward_data.smooth_barrier_energy}

        if fit_normal:
            # self.transform_pos(target["mvp"], opt_mesh.v_nrm, is_vec=True)
            v_s = geometry_forward_data._compute_vertex_normal()[None, ...]
            scale = torch.tensor([1, 1, -1], dtype=torch.float32,
                                 device=self.device)[None, None, :]  # for wonder 3d gso
            # scale = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)[None, None, :]

            v_s *= scale
            v_n, _ = dr.interpolate(
                v_s, rast_out, geometry_forward_data.t_pos_idx)
            # v_n = torch.cat((v_n[..., 0:3], alpha), dim=-1)

            out["n"] = v_n

        if fit_depth:
            assert campos is not None
            world_pos, _ = dr.interpolate(
                geometry_forward_data.v_pos[None, ...], rast_out, geometry_forward_data.t_pos_idx)
            camera_pos = campos[:, None, None, :]

            diff = torch.norm(world_pos - camera_pos, dim=-1, keepdim=True)
            # d = torch.cat((diff, alpha), dim=-1)

            d = diff
            out["d"] = d

        return out

    def export(self, path: str, folder: str, texture_res: int = 1024):
        assert self.materials is not None

        os.makedirs(os.path.join(path, folder), exist_ok=True)

        v_pos = self.geometry.tet_v.clone().detach()[self.geometry.surface_vid]
        t_pos_idx = self.geometry.surface_fid
        v_tex = self.geometry.uv
        t_tex_idx = self.geometry.uv_idx

        # uv_clip = v_tex * 2 - 1
        # uv_clip4 = torch.cat([uv_clip,
        #                       torch.zeros_like(uv_clip[..., 0:1]),
        #                       torch.ones_like(uv_clip[..., 0:1])],
        #                      dim=-1)

        # rast, _ = dr.rasterize(self.glctx, uv_clip4[None, ...], t_pos_idx, resolution=[
        #                        texture_res, texture_res])
        # hole_mask = ~(rast[0, :, :, 3] > 0)

        # def uv_padding(image):
        #     uv_padding_size = 2
        #     inpaint_image = (
        #         cv2.inpaint(
        #             (image.detach().cpu().numpy() * 255).astype(np.uint8),
        #             (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
        #             uv_padding_size,
        #             cv2.INPAINT_TELEA,
        #         )
        #     )
        #     return inpaint_image

        # gb_pos, _ = dr.interpolate(v_pos[None, ...], rast, t_pos_idx)
        # gb_pos = gb_pos[0]

        # mat_out = self.materials.forward(gb_pos)
        # save_mat = uv_padding(mat_out["color"].clone())
        # import pdb; pdb.set_trace()
        # # cv2.imwrite(os.path.join(path, folder, "exported_surface.png"), save_mat)

        # _save_obj(path=os.path.join(path, folder),
        #           filename="exported_surface.obj",
        #           v_pos=v_pos.cpu().numpy(),
        #           t_pos_idx=t_pos_idx.cpu().numpy(),
        #           v_tex=v_tex.cpu().numpy(),
        #           t_tex_idx=t_tex_idx.cpu().numpy(),
        #           matname="exported_surface",

        #           mtllib="exported_surface.mtl")
        # _save_mtl(path=os.path.join(path, folder),
        #           filename="exported_surface.mtl",
        #           map_Kd=save_mat,
        #           matname="exported_surface",
        #           map_format="png")
        
        with torch.no_grad():
            mat_out = self.materials.forward(v_pos)
        ## export surface obj with vertex color
        surface_mesh = trimesh.Trimesh(vertices=v_pos.cpu().numpy(), faces=t_pos_idx.cpu().numpy())
        surface_mesh.visual.vertex_colors = mat_out["color"].cpu().numpy()
        surface_mesh.export(os.path.join(path, folder, "exported_surface.obj"))

        # use PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(path, folder, "exported_surface.obj"))

        # Check if the mesh has vertex colors
        if not ms.current_mesh().has_vertex_color():
            print("The mesh does not have vertex colors.")
            exit(0)

        # Create UV coordinates using a trivial per-triangle parameterization
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(border=1)
        # Bake vertex colors to a texture
        ms.transfer_attributes_to_texture_per_vertex(textw=texture_res, texth=texture_res)
        # Save the mesh with the new UV coordinates
        ms.save_current_mesh(os.path.join(path, folder,"exported_surface.obj"), save_textures=True)
