











from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

from .utils.segmentation_processing import (erase_small_nuclei, erase_solitary,
                                            stitch3D_z)


def folder_segmentation(path_to_staining,
                        regex_staining,
                        path_to_mask,
                        dico_param,
                        output_dtype=np.int32
                        ):
    from cellpose import models ## speed import

    """
    segment nuclei or cytoplasms  and save them in th path_to_mask_dapi folder
    Args:
        path_to_dapi (str): path to the folder containing the dapi images
        path_to_mask_dapi (str):  path to the folder where the mask will be saved
        dico_param (dict): dictionary with the cell pose parameters
        model (cellpose model): cellpose model
        save (bool):
        output_dtype (np.dtype): dtype of the mask
    Returns:
        None
    """
    Path(path_to_mask).mkdir(parents=True, exist_ok=True)
    model = models.Cellpose(gpu=dico_param['gpu'], model_type=dico_param['model_type'])
    if path_to_mask[-1] != "/":
        path_to_mask += "/"
    if path_to_staining[-1] != "/":
        path_to_staining += "/"
    print(list(Path(path_to_staining).glob(f"{regex_staining}")))
    print(f'dico_param{dico_param}')
    for path_dapi in tqdm(list(Path(path_to_staining).glob(f"{regex_staining}"))):
        path_dapi = str(path_dapi)
        print(path_dapi)
        img = tifffile.imread(path_dapi)
        print(img.shape)
        if dico_param["mip"] is True and len(img.shape) == 3:
            img = np.amax(img, 0)
        else:
            if len(img.shape) == 3:
                img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
                print(f'image dapi shape after reshape {img.shape}')
                img = list(img)
        masks, flows, styles, diams = model.eval(img,
                                                 diameter=dico_param["diameter"],
                                                 channels=[0, 0],
                                                 flow_threshold=dico_param["flow_threshold"],
                                                 do_3D=dico_param["do_3D"],
                                                 stitch_threshold=0)

        masks = np.array(masks, dtype=output_dtype)
        if len(masks.shape) == 3:
            masks = stitch3D_z(masks, dico_param["stitch_threshold"])
            masks = np.array(masks, dtype = output_dtype)
            if len(masks.shape) and dico_param["erase_solitary"]:
                masks = erase_solitary(masks)
        if dico_param["erase_small_nuclei"] is not None:
            print(f'erase_small_nuclei threshold {dico_param["erase_small_nuclei"]}')
            masks = erase_small_nuclei(masks)
        image_name = path_dapi.split('/')[-1].split(f'_{regex_staining}')[0]
        tifffile.imwrite(path_to_mask + image_name +'.tif', data=masks, dtype=masks.dtype)
        np.save(path_to_mask + "dico_param.npy", dico_param)



if __name__ == "__main__":
    from cellpose import models  ## speed import

    model = models.Cellpose(gpu=dico_param['gpu'], model_type=dico_param['model_type'])

    masks, flows, styles, diams = model.eval(img,
                                             diameter=dico_param["diameter"],
                                             channels=[0, 0],
                                             flow_threshold=dico_param["flow_threshold"],
                                             do_3D=dico_param["do_3D"],
                                             stitch_threshold=0)



