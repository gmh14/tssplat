import numpy as np
import torch


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


def compute_G_matrix(verts_init, faces):
    # compute gradient operator matrix
    T = faces.shape[0]

    if type(verts_init) == np.ndarray:
        verts_init = torch.from_numpy(verts_init).to(torch.float64)

    Gd = torch.zeros([4, 3], device=verts_init.device).to(torch.float64)
    Gd[0, :] = -1.0
    Gd[1, 0] = 1.0
    Gd[2, 1] = 1.0
    Gd[3, 2] = 1.0  # 4 x 3

    X = verts_init[faces]  # T x 4 x 3
    X = X.transpose(1, 2)  # T x 3 x 4
    dX = torch.matmul(X, Gd.unsqueeze(0).expand(T, -1, -1))  # T x 3 x 3

    dX_inv = torch.inverse(dX)  # T x 3 x 3

    # G = torch.matmul(Gd.unsqueeze(0).expand(T, -1, -1), dX_inv) # T x 4 x 3
    # return G

    G = torch.zeros([T, 9, 12], device=verts_init.device).to(torch.float64)
    for dofi in range(12):
        E = torch.zeros([3, 4], device=verts_init.device).to(torch.float64)
        E[dofi % 3, dofi // 3] = 1.0

        Z = E @ Gd
        R = Z.unsqueeze(0).expand(T, -1, -1) @ dX_inv
        G[:, :, dofi] = R.view(T, -1)

    return G.numpy()  # T x 9 x 12
