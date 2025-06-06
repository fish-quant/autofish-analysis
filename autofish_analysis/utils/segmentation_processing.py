

import cellpose
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm

def stitch3D_z(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        try:
            iou = cellpose.metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
        except Exception as e:
            print(e)
            print("in stich")
            continue
    return masks



def erase_solitary(mask): #mask en 3D
    """
    Erase nuclei  that are present in only one Z-slice
    Args:
        mask ():

    Returns:

    """
    mask_bis = np.zeros(mask.shape)
    current_nuclei = set(np.unique(mask[0]))
    post_nuclei = set(np.unique(mask[1]))
    nuclei_to_remove =  current_nuclei - post_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[0] += (mask[0] == nuc) * mask[0]

    for i in range(1, len(mask)-1):
        pre_nuclei = set(np.unique(mask[i-1]))
        current_nuclei = set(np.unique(mask[i]))
        post_nuclei = set(np.unique(mask[i+1]))
        nuclei_to_remove =  current_nuclei - pre_nuclei - post_nuclei
        nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
        for nuc in nuclei_to_keep:
            mask_bis[i] += (mask[i] == nuc) *  mask[i]
    ##traiter le cas ou n = -1
    current_nuclei = set(np.unique(mask[-1]))
    pre_nuclei = set(np.unique(mask[-2]))
    nuclei_to_remove =  current_nuclei - pre_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[-1] += (mask[-1] == nuc) * mask[-1]
    return mask_bis




def erase_small_nuclei(mask, min_size = 340):
    for nuc in tqdm(np.unique(mask)[1:]): ## remove zero
        sum_size = np.sum((mask == nuc).astype(int))
        print(sum_size)
        if sum_size < min_size:
                mask[mask == nuc] = 0
    return mask



def compute_dico_centroid(mask_nuclei, dico_simu = None, offset = np.array([0,0,0])):
    from skimage import measure
    dico_nuclei_centroid = {}
    #nuclei_labels = measure.label(mask_nuclei, background=0)
    for lb in measure.regionprops(mask_nuclei):
        assert lb.label != 0
        dico_nuclei_centroid[lb.label] = {}
        dico_nuclei_centroid[lb.label]['centroid'] = np.array(lb.centroid) + offset
        #print(lb.centroid)
        if dico_simu is not None:
            dico_nuclei_centroid[lb.label]['type'] = dico_simu['dico_cell_index'][lb.label]['type']
    return dico_nuclei_centroid




########## generate artefact mask ##########



def get_artefact_dico_percent(round = "r1",
            pos = "pos4",
            image_shape = (40, 2048, 2048),
            mask = None, #tifffile.imread(f"/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/segmentation_mask/r1_pos4.tif"),
            dico_spot_artefact = None,   #np.load(f"/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_spot_artefact.npy",
                                    #     allow_pickle=True).item(),
                              ):


    mask_log_3D = np.zeros(image_shape)

    array_spots_artefact = dico_spot_artefact[round][pos]
    mask_coord = array_spots_artefact[:, :].astype(int)
    valid_coord = np.logical_and(np.logical_and(mask_coord[:, 1] < image_shape[1],
                                                mask_coord[:, 2] < image_shape[2]),
                                 np.logical_and(mask_coord[:, 1] > 0, mask_coord[:, 2] > 0))
    mask_coord = mask_coord[valid_coord, :]
    mask_log_3D[mask_coord[:, 0],mask_coord[:, 1], mask_coord[:, 2]] = 1
    mask_log_3D = ndimage.maximum_filter(mask_log_3D, size=[20, 30, 30])


    unique_nuc = np.unique(mask)[1:]
    set_art = set(list(zip(*np.nonzero(mask_log_3D))))
    dico_per_art = {}
    for nuc in tqdm(unique_nuc):
        set_nuc = set(list(zip(*np.nonzero(mask==nuc))))

        dico_per_art[nuc] = len(set_art.intersection(set_nuc)) / len(set_nuc)
    return dico_per_art

def get_artefact_dico_percent_folder(ref_round,
                                 path_seg_mask,
                                 dico_spot_artefact,
                        image_shape = (40, 2048, 2048),):

    dico_per_art = {}
    for path_ind_mask in tqdm(list(Path(path_seg_mask).glob('*tif'))[:]):
        pos = "pos" +  str(path_ind_mask).split("/")[-1].split("pos")[1].split(".")[0]
        mask = tifffile.imread(path_ind_mask)

        dico_per_art[pos] = get_artefact_dico_percent(round = ref_round,
            pos = pos,
            image_shape = (40, 2048, 2048),
            mask =mask,
            dico_spot_artefact = dico_spot_artefact)
    return dico_per_art





if False:

    import napari
    viewer = napari.Viewer()
    viewer.add_image(mask_log_3D)
    viewer.add_image(mask)

    np.save("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/dico_per_art.npy", dico_per_art)
    dico_per_artdico_per_art = np.load('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/dico_spot_artefact.npy',
                                        allow_pickle=True).item()

    dico_new_to_old_label_pos = np.load('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_new_to_old_label_pos.npy',
                                        allow_pickle=True).item()
    dico_spot_artefact = np.load('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_spot_artefact.npy',
                                 allow_pickle=True).item()
    dico_per_art  =  get_artefact_dico_percent_folder(ref_round = 'r1',
                                     path_seg_mask = "/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/segmentation_mask/",
                                     dico_spot_artefact = dico_spot_artefact ,
                                     image_shape=(40, 2048, 2048), )


    np.save("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/dico_per_art.npy", dico_per_art)

    ################# inverse dico_new_to_old_label_pos
    dico_old_to_new_label_pos = {}

    for pos in dico_new_to_old_label_pos:
        dico_old_to_new_label_pos[pos] = {}
        for new  in dico_new_to_old_label_pos[pos]:
            dico_old_to_new_label_pos[pos][dico_new_to_old_label_pos[pos][new]] = new

    list_macro_phage = []

    for pos in dico_per_art:
        for nuc in dico_per_art[pos]:
            if dico_per_art[pos][nuc] > 0.5:
                try:
                    list_macro_phage.append(dico_old_to_new_label_pos[pos][nuc])
                except KeyError:
                    print(nuc)
                    pass
    np.save('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/list_macrophage', list_macro_phage)


## classification of artefact



















