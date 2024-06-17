from .dataloader import Wonder3DDataLoader, MistubaImgDataLoader, BlenderImgDataLoader


def load_dataloader(dataloader_class_type):
    if dataloader_class_type == "Wonder3DDataLoader":
        return Wonder3DDataLoader
    elif dataloader_class_type == "MistubaImgDataLoader":
        return MistubaImgDataLoader
    elif dataloader_class_type == "BlenderImgDataLoader":
        return BlenderImgDataLoader
    else:
        raise NotImplementedError(
            f"Unknown data class type: {dataloader_class_type}")
