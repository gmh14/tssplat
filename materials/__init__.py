from .explicit_material import ExplicitMaterial


def load_material(material_class_type):
    if material_class_type == "ExplicitMaterial":
        return ExplicitMaterial
    else:
        raise NotImplementedError(
            f"Unknown geometry class type: {material_class_type}")
