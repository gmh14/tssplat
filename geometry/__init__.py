from .tetmesh_geometry import TetMeshGeometry, TetMeshMultiSphereGeometry
from .tetmesh_fish import TetMeshFish


def load_geometry(geometry_class_type):
    if geometry_class_type == "TetMeshFish":
        return TetMeshFish
    elif geometry_class_type == "TetMeshMultiSphereGeometry":
        return TetMeshMultiSphereGeometry
    else:
        raise NotImplementedError(
            f"Unknown geometry class type: {geometry_class_type}")
