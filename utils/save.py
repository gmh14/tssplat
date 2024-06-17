import numpy as np
import os
import cv2

from .typing import *


def _save_obj(
    path,
    filename,
    v_pos,
    t_pos_idx,
    v_nrm=None,
    v_tex=None,
    t_tex_idx=None,
    v_rgb=None,
    matname=None,
    mtllib=None,
) -> str:
    obj_str = ""
    if matname is not None:
        obj_str += f"mtllib {mtllib}\n"
        obj_str += f"g object\n"
        obj_str += f"usemtl {matname}\n"
    for i in range(len(v_pos)):
        obj_str += f"v {v_pos[i][0]} {v_pos[i][1]} {v_pos[i][2]}"
        if v_rgb is not None:
            obj_str += f" {v_rgb[i][0]} {v_rgb[i][1]} {v_rgb[i][2]}"
        obj_str += "\n"
    if v_nrm is not None:
        for v in v_nrm:
            obj_str += f"vn {v[0]} {v[1]} {v[2]}\n"
    if v_tex is not None:
        for v in v_tex:
            obj_str += f"vt {v[0]} {1.0 - v[1]}\n"

    for i in range(len(t_pos_idx)):
        obj_str += "f"
        for j in range(3):
            obj_str += f" {t_pos_idx[i][j] + 1}/"
            if v_tex is not None:
                obj_str += f"{t_tex_idx[i][j] + 1}"
            obj_str += "/"
            if v_nrm is not None:
                obj_str += f"{t_pos_idx[i][j] + 1}"
        obj_str += "\n"

    save_path = os.path.join(path, filename)
    with open(save_path, "w") as f:
        f.write(obj_str)
    return save_path


def _save_mtl(
    path,
    filename,
    matname,
    Ka=(0.0, 0.0, 0.0),
    Kd=(1.0, 1.0, 1.0),
    Ks=(0.0, 0.0, 0.0),
    map_Kd=None,
    map_Ks=None,
    map_Bump=None,
    map_Pm=None,
    map_Pr=None,
    map_format="jpg",
    step: Optional[int] = None,
) -> List[str]:
    mtl_save_path = os.path.join(path, filename)

    save_paths = [mtl_save_path]

    mtl_str = f"newmtl {matname}\n"
    mtl_str += f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
    if map_Kd is not None:
        map_Kd_save_path = os.path.join(
            os.path.dirname(mtl_save_path), f"texture_kd.{map_format}"
        )
        mtl_str += f"map_Kd texture_kd.{map_format}\n"
        cv2.imwrite(map_Kd_save_path, map_Kd)  # rgb

        save_paths.append(map_Kd_save_path)
    else:
        mtl_str += f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"

    if map_Ks is not None:
        map_Ks_save_path = os.path.join(
            os.path.dirname(mtl_save_path), f"texture_ks.{map_format}"
        )
        mtl_str += f"map_Ks texture_ks.{map_format}\n"
        cv2.imwrite(map_Ks_save_path, map_Ks)  # rgb
        save_paths.append(map_Ks_save_path)
    else:
        mtl_str += f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"

    if map_Bump is not None:
        map_Bump_save_path = os.path.join(
            os.path.dirname(mtl_save_path), f"texture_nrm.{map_format}"
        )
        mtl_str += f"map_Bump texture_nrm.{map_format}\n"
        cv2.imwrite(map_Bump_save_path, map_Bump)  # rgb
        save_paths.append(map_Bump_save_path)

    if map_Pm is not None:
        map_Pm_save_path = os.path.join(
            os.path.dirname(mtl_save_path), f"texture_metallic.{map_format}"
        )
        mtl_str += f"map_Pm texture_metallic.{map_format}\n"
        cv2.imwrite(map_Pm_save_path, map_Pm)  # gray scale
        save_paths.append(map_Pm_save_path)

    if map_Pr is not None:
        map_Pr_save_path = os.path.join(
            os.path.dirname(mtl_save_path), f"texture_roughness.{map_format}"
        )
        mtl_str += f"map_Pr texture_roughness.{map_format}\n"
        cv2.imwrite(map_Pr_save_path, map_Pr)  # gray scale
        save_paths.append(map_Pr_save_path)

    with open(mtl_save_path, "w") as f:
        f.write(mtl_str)

    return save_paths
