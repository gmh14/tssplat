import pypgo

import os
import sys
import numpy as np
from dataclasses import dataclass, field

import trimesh
import xatlas

from .mesh_utils import get_surface_vf


def load_tet_mesh_from_veg_file(filename):
    tetmesh = pypgo.create_tetmesh_from_file(filename)
    vtx_init = pypgo.get_tetmesh_vertex_positions(tetmesh)
    elem = pypgo.get_tetmesh_element_indices(tetmesh)
    return vtx_init, elem, tetmesh


def loat_tet_mesh_from_np_files(tet_vtx, tet_elem, E, nu, density):
    tetmesh = pypgo.create_tetmesh(tet_vtx.flatten().astype(
        np.float32), tet_elem.flatten().astype(np.int32), E, nu, density)
    return tet_vtx, tet_elem, tetmesh


class TetrahedronMesh:

    def __init__(self, **kwargs):
        self.density = 1000
        self.E = 100000
        self.nu = 0.45

        if "veg_file_path" in kwargs:
            # load from veg
            self.vtx_init, self.elem, self.tetmesh = load_tet_mesh_from_veg_file(
                kwargs["veg_file_path"])

        elif "surf_mesh_file_path" in kwargs:
            pass  # load from surface mesh

        elif "vtx_np_file_path" in kwargs:
            assert kwargs["elem_np_file_path"]
            tet_vtx = np.load(kwargs["vtx_np_file_path"])
            tet_elem = np.load(kwargs["elem_np_file_path"])
            self.vtx_init, self.elem, self.tetmesh = loat_tet_mesh_from_np_files(
                tet_vtx, tet_elem, self.E, self.nu, self.density)

        elif "vtx_np" in kwargs:
            assert "elem_np" in kwargs
            self.vtx_init, self.elem, self.tetmesh = loat_tet_mesh_from_np_files(
                kwargs["vtx_np"], kwargs["elem_np"], self.E, self.nu, self.density)

        else:
            raise ValueError(
                "Either veg_file_path or surf_mesh_file_path should be provided.")

        if "surface_vid_np" in kwargs and "surface_fid_np" in kwargs:
            self.surface_vid = kwargs["surface_vid_np"]
            self.surface_fid = kwargs["surface_fid_np"]
        else:
            self.surface_vid, self.surface_fid = get_surface_vf(self.elem)

        self.vtx = self.vtx_init.copy()

        surface_v = self.vtx[self.surface_vid]
        vmapping, self.uv_idx, self.uv = xatlas.parametrize(
            surface_v, self.surface_fid)

    def update_vtx_pos(self, vtx: np.ndarray):
        self.vtx = vtx.copy()
        tetmesh_new = pypgo.update_tetmesh_vertices(self.tetmesh, vtx)
        self.tetmesh = tetmesh_new

    def save_surface_mesh(self, path: str, filename: str = "surface_mesh.obj"):
        os.makedirs(path, exist_ok=True)
        surface_vtx = self.vtx[self.surface_vid].copy()
        surface_faces = self.surface_fid.copy()
        mesh = trimesh.Trimesh(vertices=surface_vtx, faces=surface_faces)
        mesh.export(os.path.join(path, filename))

    def save(self, path: str, filename: str = "tet_mesh", save_surface_mesh: bool = True, save_npy: bool = False):
        os.makedirs(path, exist_ok=True)
        pypgo.save_tetmesh_to_file(
            self.tetmesh, os.path.join(path, filename + ".veg"))

        if save_surface_mesh:
            self.save_surface_mesh(path, filename + "_surface_mesh.obj")
        if save_npy:
            np.save(os.path.join(path, filename + "_vtx.npy"), self.vtx)
            np.save(os.path.join(path, filename + "_elem.npy"), self.elem)
