import mitsuba as mi
import numpy as np
import math
import os
import trimesh
import sys

print(mi.variants())
mi.set_variant("cuda_ad_rgb")
res = 512
fov = 39.3077
near = 0.001
far = 10
aspect = 1


def look_at(eye, center, up):
    lookat = (center - eye) / np.linalg.norm(center - eye)

    right = np.cross(lookat, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, lookat)
    up /= np.linalg.norm(up)

    M = np.eye(4)
    M[0, 0:3] = right
    M[1, 0:3] = up
    M[2, 0:3] = -lookat

    M[0, 3] = -np.dot(right, eye)
    M[1, 3] = -np.dot(up, eye)
    M[2, 3] = np.dot(lookat, eye)

    return M


def perspective():
    tan_half_fovy = math.tan(fov / 180.0 * math.pi * 0.5)
    M = np.zeros((4, 4), dtype=np.float32)

    M[0, 0] = 1 / (aspect * tan_half_fovy)
    M[1, 1] = -1 / tan_half_fovy
    M[2, 2] = -(far + near) / (far - near)
    M[3, 2] = -1
    M[2, 3] = -(2 * far * near) / (far - near)

    return M


def load_sensor(origin, target, up):
    v = look_at(origin, target, up)
    p = perspective()

    v1 = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)

    # print(v)
    # print(v1)

    sensor = mi.load_dict(
        {
            "type": "perspective",
            "fov": fov,
            "near_clip": near,
            "far_clip": far,
            "to_world": v1,
            "sampler": {"type": "independent", "sample_count": 16},
            "film": {
                "type": "hdrfilm",
                "width": res,
                "height": res,
                "rfilter": {
                    "type": "tent",
                },
                "pixel_format": "rgba",
            },
        }
    )

    mvp = p @ v
    return sensor, mvp, v, origin


def convert_transform2numpy(transform) -> np.ndarray:
    matrix = np.zeros([4, 4], dtype=float)
    for i in range(4):
        for j in range(4):
            matrix[i, j] = transform.matrix[i, j]
    return matrix


def sample_view(r, n_total):
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, n_total)
    theta = 2 * math.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i) / n_total)
    x, y, z = np.cos(theta) * np.sin(phi) * r, np.sin(theta) * \
        np.sin(phi) * r, np.cos(phi) * r

    sensors = []
    mvps = []
    mvs = []
    eyes = []
    for fi in range(n_total):
        cx = x[fi]
        cy = y[fi]
        cz = z[fi]

        eye = np.asarray([cx, cy, cz])
        center = np.zeros(3)
        diff = eye - center
        diff /= np.linalg.norm(diff)

        up = np.asarray([0, 0, 1])

        if abs(np.dot(up, diff)) > math.cos(math.pi / 8.0):
            up = (0, 1, 0)

        sensor, mvp, mv, origin = load_sensor(eye, center, up)

        if isinstance(mv, np.ndarray):
            mvs.append(mv)
        else:
            mvs.append(convert_transform2numpy(mv))

        eyes.append(origin)

        if isinstance(mvp, np.ndarray):
            mvps.append(mvp)
        else:
            mvps.append(convert_transform2numpy(mvp))

        sensors.append(sensor)

    return sensors, mvps, mvs, eyes


def render_mesh(base_path, filename, tex_filename, n_total):
    mesh_filename = os.path.join(base_path, filename)
    tex_filename = os.path.join(base_path, tex_filename)

    scene = mi.load_dict(
        {
            "type": "scene",
            # The keys below correspond to object IDs and can be chosen arbitrarily
            "integrator": {
                "type": "aov",
                "aovs": "dd:depth,nn:sh_normal",
                "color": {
                    "type": "path",
                    "hide_emitters": True,
                },
            },
            "light": {"type": "constant"},
            "teapot": {
                "type": "obj",
                "filename": mesh_filename, # "temp.obj",
                "to_world": mi.ScalarTransform4f.scale([1, 1, 1]),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": tex_filename,
                    },
                },
            },
        }
    )

    (
        sensors,
        mvps,
        mvs,
        eyes,
    ) = sample_view(4, n_total)
    images = []

    inc = 0
    for sensor in sensors:
        film = sensor.film()
        scene.integrator().render(scene, spp=16, sensor=sensor)

        layers = film.bitmap().split()
        images.append(layers)

        print(inc, end=" ")
        sys.stdout.flush()

        inc += 1

    print()

    return images, mvps, mvs, eyes

def main(base_folder, mesh, texture, target_folder_dense, f):
    # normalize fish
    mesh_data = trimesh.load(os.path.join(base_folder, mesh))
    if isinstance(mesh_data, trimesh.Scene):
        if len(mesh_data.geometry) == 0:
            mesh_data = None  # empty scene
        else:
            # we lose texture information here
            mesh_data = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh_data.geometry.values()))
    else:
        assert(isinstance(mesh_data, trimesh.Trimesh))
        
    v_center = np.mean(mesh_data.vertices, axis=0)
    mesh_data.vertices -= v_center

    v_min = np.min(mesh_data.vertices, axis=0)
    v_max = np.max(mesh_data.vertices, axis=0)

    v_center = (v_max + v_min) * 0.5
    bb_size = np.max(v_max - v_center)
    scale = 1.0 / bb_size

    mesh_data.vertices = (mesh_data.vertices - v_center) * scale # + v_center
    mesh_data.export(os.path.join(base_folder, "{}_normalized.obj".format(os.path.basename(mesh).split(".")[0])))

    # use PyMeshLab
    images, mvps, mvs, eyes = render_mesh(
        base_folder, "{}_normalized.obj".format(os.path.basename(mesh).split(".")[0]), texture, 120)

    os.makedirs(os.path.join(target_folder_dense, f), exist_ok=True)
    for i, img in enumerate(images):
        # img = img ** (1.0 / 2.2)
        # img = mi.Bitmap(img, mi.Bitmap.PixelFormat.RGBA)
        for name, pic in img:
            if name == "color":
                color = pic.convert(
                    pixel_format=mi.Bitmap.PixelFormat.RGBA, component_format=mi.Struct.Type.UInt8, srgb_gamma=True
                )
                color.write(os.path.join(
                    target_folder_dense, f, f"img_rgba_{i}.png"))
            elif name == "dd":
                mp_np = np.array(pic)
                np.save(os.path.join(target_folder_dense,
                        f, f"depth_{i}.npy"), mp_np)
            elif name == "nn":
                color = pic.convert(
                    pixel_format=mi.Bitmap.PixelFormat.RGBA, component_format=mi.Struct.Type.UInt16, srgb_gamma=False
                )
                color.write(os.path.join(target_folder_dense,
                            f, f"normal_rgba_{i}.png"))

                mp_np = np.array(pic)
                np.save(os.path.join(target_folder_dense,
                        f, f"normal_{i}.npy"), mp_np)

        # Write image to JPEG file

        np.save(os.path.join(target_folder_dense,
                f, f"mvp_mtx_{i}.npy"), mvps[i])
        np.save(os.path.join(target_folder_dense,
                f, f"mv_{i}.npy"), mvs[i])


if __name__ == "__main__":
    base_folder = "../mesh_data/mario_example"
    mesh = "model.obj"
    texture = "texture.png"
    target_folder_dense = "../img_data/"
    folder = "mario"
    main(base_folder, mesh, texture, target_folder_dense, folder)
    