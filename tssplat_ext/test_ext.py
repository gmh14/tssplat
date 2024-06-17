try:
    import torch
    from tet_spheres import tet_spheres_ext  # NOQA
    print("tet_spheres_ext imported successfully")
except:
    # print("using PyTorch implementation of tet_spheres_ext")
    print("error importing tet_spheres_ext")
    