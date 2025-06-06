

import itertools
import re
from pathlib import Path

import bigfish.detection as detection
import bigfish.stack as stack
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from .utils.signal_quality import mean_cos_tetha


def remove_artifact(filtered_fish, spots, order = 3, min_cos_tetha = 0.75):

    gz, gy, gx = np.gradient(filtered_fish)
    real_spots = []
    for s in spots:
        if mean_cos_tetha(gz, gy, gx,
                          z=s[0], yc=s[1],
                          xc=s[2], order=order) > min_cos_tetha:
            real_spots.append(s)
    return real_spots



def remove_double_detection(input_array,
            threshold = 0.3,
            scale_z_xy = np.array([0.300, 0.103, 0.103])):
    """
    Args:
        input_list (np.array):
        threshold (float): min distance between point in um
        scale_z_xy (np.array):voxel scale in um
    Returns: list of point without double detection
    """
    unique_tuple = [tuple(s) for s in input_array]
    unique_tuple = list(set((unique_tuple)))

    combos = itertools.combinations(unique_tuple, 2)
    points_to_remove = [list(point2)
                        for point1, point2 in combos
                        if np.linalg.norm(point1 * scale_z_xy  - point2 * scale_z_xy) < threshold]

    points_to_keep = [list(point) for point in unique_tuple if list(point) not in points_to_remove]
    return points_to_keep






def detection_with_segmentation(rna,
                                sigma,
                                min_distance = [3,3, 3],
                              segmentation_mask = None,
                              diam_um = 20,
                              scale_xy = 0.103,
                              scale_z = 0.300,
                              min_cos_tetha = 0.75,
                              order = 5,
                              test_mode = False,
                              threshold_merge_limit = 0.330,
                              x_translation_mask_to_rna = 0,
                              y_translation_mask_to_rna = 0,
                              use_median_threshold = False,
                            remove_non_sym = True):


    """
    Args:
        rna (np.array): fish image
        sigma (): sigma in gaussian filter
        min_distance (): big fish parameter
        segmentation_mask (np.array):  nuclei segmentation
        diam (): radias size in um around nuclei where to detect spots
        scale_xy (): pixel xy size in um

        scale_z (): pixel e size in um
        min_cos_tetha (float): value between 0 and 1, if 0 it does not remove anything, if 1 it accept only perfect radial symetric spots
        order (): radias size in pixel around spots where to check radail symetrie spts
        threshold_merge_limit (float): threshold below to detected point are considere the same
    Returns:
    """

    rna_log = stack.log_filter(rna, sigma)
    mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    rna_gaus = ndimage.gaussian_filter(rna, sigma)

    list_of_nuc = np.unique(segmentation_mask)
    if 0 in list_of_nuc:
        list_of_nuc = list_of_nuc[1:]
    assert all(i >= 1 for i in list_of_nuc)

    all_spots = []
    pbar = tqdm(list_of_nuc)
    threshold_list = []
    for mask_id in pbar:
        try:
            pbar.set_description(f"detecting rna around cell {mask_id}")
            [Zm,Ym, Xm] = ndimage.center_of_mass(segmentation_mask == mask_id)
            Xm -= x_translation_mask_to_rna
            Ym -= y_translation_mask_to_rna
            Y_min = np.max([0, Ym - diam_um / scale_xy]).astype(int)
            Y_max = np.min([segmentation_mask.shape[1], Ym + diam_um / scale_xy]).astype(int)
            X_min = np.max([0, Xm - diam_um / scale_xy]).astype(int)
            X_max = np.min([segmentation_mask.shape[2], Xm + diam_um / scale_xy]).astype(int)
            crop_mask = mask[:, Y_min:Y_max, X_min:X_max]
            threshold = detection.automated_threshold_setting(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask)
            threshold_list.append(threshold)
            if use_median_threshold:
                print(f"using median threshold {threshold}")
                continue
            spots, _ = detection.spots_thresholding(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask, threshold)

            if remove_non_sym:
                new_spots = remove_artifact(filtered_fish = rna_gaus[:,Y_min:Y_max, X_min:X_max],
                                spots = spots,
                                order=order,
                                min_cos_tetha=min_cos_tetha)
                """new_spots = []
                for s in spots:
                    if mean_cos_tetha(filtered_crop_fish = rna_gaus[:,Y_min:Y_max, X_min:X_max],
                                      z=s[0], yc=s[1],
                                      xc=s[2], order=order) > min_cos_tetha:
                        new_spots.append(s)"""

            if test_mode: ## test mode
                input = np.amax(rna[:, Y_min:Y_max, X_min:X_max], 0)
                pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
                rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
                fig, ax = plt.subplots(2, 2, figsize=(30, 30))
                plt.title(f' X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}', fontsize=20)

                ax[0, 0].imshow(rna_scale)
                ax[0, 1].imshow(rna_scale)
                for s in spots:
                    ax[0, 0].scatter(s[-1], s[-2], c='red', s=28)
                # plt.show()

                # fig, ax = plt.subplots(2, 1, figsize=(15, 15))
                plt.title(
                    f'with remove artf  order{order}  min_tetha {min_cos_tetha}  X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}',
                    fontsize=20)
                ax[1, 0].imshow(rna_scale)
                ax[1, 1].imshow(rna_scale)
                for s in new_spots:
                    ax[1, 0].scatter(s[-1], s[-2], c='red', s=28)
                plt.show()
            spots = new_spots
            spots = np.array(spots)
            if len(spots) > 0:
                spots = spots + np.array([0, Y_min, X_min])
                all_spots += list(spots)
        except Exception as e:
            print(e)
            print(f"error in cell {mask_id}")
            continue

    if use_median_threshold:
        print("median")
        threshold = np.median(threshold_list)
        all_spots, _ = detection.spots_thresholding(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask, threshold)
    else:
        all_spots = remove_double_detection(
                    input_array = np.array(all_spots),
                    threshold =threshold_merge_limit,
                    scale_z_xy = np.array([scale_z, scale_xy, scale_xy]))


    if test_mode:
        input = np.amax(rna, 0)
        pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
        rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
        fig, ax = plt.subplots(2, 1, figsize=(40, 40))
        ax[0].imshow(rna_scale)
        ax[1].imshow(rna_scale)
        for s in all_spots:
            ax[0].scatter(s[-1], s[-2], c='red', s=28)
        plt.show()

    return all_spots


def detection_without_segmentation(
                            rna,
                            sigma,
                            min_distance = [3,3, 3],
                            threshold = None,
                            mask_log_2D = None,
                            min_cos_tetha=0.50,
                            order=5,
                            remove_non_sym = False):


    rna_log = stack.log_filter(rna, sigma)
    if mask_log_2D is not None:
        rna_log[:, mask_log_2D == 1] = 0
        mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    else:
        mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    if threshold is None:
        threshold = detection.automated_threshold_setting(rna_log, mask)
    spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
    if remove_non_sym:
        print(f'removing non symetric spots with min cos tetha {min_cos_tetha}')
        rna_gaus = ndimage.gaussian_filter(rna, sigma)

        new_spots = remove_artifact(filtered_fish=rna_gaus,
                                    spots=spots,
                                    order=order,
                                    min_cos_tetha=min_cos_tetha)
        return  new_spots, threshold

    return spots, threshold

#%%

def folder_detection(
                     folder_of_rounds = "/media/tom/T7/Stitch/acquisition/",
                     round_name_regex = "r",
                     image_name_regex = "opool1_1_MMStack_3",
                     channel_name_regex = "ch1",
                      min_distance=(4, 4, 4),
                      scale_xy=0.103,
                      scale_z=0.300,
                      sigma = 1.35,
                    path_output_segmentaton=None,
                    dico_spot_artefact = None,
                    artefact_filter_size = 32,
                    remove_non_sym = False,
                    ### detection parameters with segmentation
                    dico_translation = None, # np.load("/media/tom/T7/Stitch/acquisition/dico_translation.npy", allow_pickle=True).item(),
                    diam_um=20,
                    local_detection = False,
                    fixed_round_name="r1_bc1",
                    min_cos_tetha=0.75,
                    order=5,
                    test_mode=False,
                    threshold_merge_limit= 0.330,
                    file_extension = ".ti",
                    threshold_input =  {},
                    use_median_threshold = False,
            ):

    """

    :param folder_of_rounds: path to the folder containing all the rounds
    :param round_name_regex: regular expression to find the round name in the folder
    :param image_name_regex: regular expression to find the image name in the folder
    :param channel_name_regex: regular expression to find the channel name in the folder
    :param min_distance:
    :param scale_xy:
    :param scale_z:
    :param sigma:
    :param path_output_segmentaton:
    :param dico_spot_artefact:
    :param artefact_filter_size:
    :param remove_non_sym:
    :param dico_translation:
    :param diam_um:
    :param local_detection:
    :param fixed_round_name: fixed round name in the folder use only if local
    :param min_cos_tetha:
    :param order:
    :param test_mode:
    :param threshold_merge_limit:
    :param file_extension:
    :param threshold_input:
    :param use_median_threshold:
    :return:
    """

    dico_spots = {}
    dico_threshold = {}


    for path_round in tqdm(list(Path(folder_of_rounds).glob(f"{round_name_regex}"))):
        print()
        print(path_round.name)
        dico_spots[path_round.name] = {}
        dico_threshold[path_round.name] = {}
        for path_rna_img in tqdm(list(path_round.glob(f"{channel_name_regex}{file_extension}*"))):

            if not re.match(image_name_regex, path_rna_img.name) :   #image_name_regex not in path_rna_img.name:

                continue
            print(path_rna_img.name)

            try:

                rna_img = tifffile.imread(path_rna_img)
                if local_detection:

                    list_path_mask = list(Path(path_output_segmentaton).glob("*"))
                    print(list_path_mask)
                    #todo : more genaral code for naming
                    image_position = "pos" +  path_rna_img.name.split('pos')[1].split('_')[0]
                    path_mask = None
                    for i in list_path_mask:
                        if "pos" in i.name:
                            if image_position == "pos" + str(i.name).split('pos')[1].split('.')[0]:
                                path_mask = i
                    if path_mask is None:
                        raise  ValueError(f"no mask found for {path_rna_img.name} at position {image_position}")

                    print(f"mask name {path_mask.name}")
                    segmentation_mask = tifffile.imread(str(path_mask))
                    ### get the translation between the mask and the rna image
                    if path_round.name == fixed_round_name:
                        x_translation_mask_to_rna = 0
                        y_translation_mask_to_rna = 0
                    else :
                        if dico_translation[image_position][fixed_round_name][path_round.name]['x_translation'] is None \
                                or dico_translation[image_position][fixed_round_name][path_round.name]['y_translation'] is None:
                            print("no translation found for round ")
                            continue
                        x_translation_mask_to_rna = dico_translation[image_position][fixed_round_name][path_round.name]['x_translation']
                        y_translation_mask_to_rna = dico_translation[image_position][fixed_round_name][path_round.name]['y_translation']



                    all_spots = detection_with_segmentation(rna = rna_img,
                                                sigma = sigma,
                                                min_distance=min_distance,
                                                segmentation_mask=segmentation_mask,
                                                diam_um=diam_um,
                                                scale_xy=scale_xy,
                                                scale_z=scale_z,
                                                min_cos_tetha=min_cos_tetha,
                                                order=order,
                                                test_mode=test_mode,
                                                threshold_merge_limit=threshold_merge_limit,
                                                x_translation_mask_to_rna=x_translation_mask_to_rna,
                                                y_translation_mask_to_rna=y_translation_mask_to_rna,
                                                use_median_threshold = use_median_threshold,
                                                remove_non_sym = remove_non_sym)
                    threshold = None


                else:
                    if threshold_input is not None and path_round.name in threshold_input.keys() and path_rna_img.name in threshold_input[path_round.name].keys():
                        threshold = threshold_input[path_round.name][path_rna_img.name]
                    else:
                        threshold = None
                    if dico_spot_artefact is not None:
                        if 'pos' in str(path_rna_img):
                            print("dico spot artefact is not None")
                            pos = "pos" + path_rna_img.name.split('pos')[1].split('_')[0]
                        else:
                            pos = list(dico_spot_artefact[path_round.name].keys())[0]
                        try:
                            assert rna_img.ndim == 3
                            array_spots_artefact = dico_spot_artefact[path_round.name][pos]
                            mask_log_2D = np.zeros(rna_img[0].shape)
                            mask_coord = array_spots_artefact[:, 1:].astype(int)
                            valid_coord = np.logical_and(np.logical_and(mask_coord[:, 0] < rna_img[0].shape[0],
                                                                        mask_coord[:, 1] < rna_img[0].shape[1]),
                                                         np.logical_and(mask_coord[:, 0] > 0, mask_coord[:, 1] > 0))
                            mask_coord = mask_coord[valid_coord, :]
                            mask_log_2D[mask_coord[:, 0], mask_coord[:, 1]] = 1
                            mask_log_2D = ndimage.maximum_filter(mask_log_2D, size = artefact_filter_size)

                        except KeyError:
                            print(f"no artefact found for {path_round.name} at position {pos}")
                            mask_log_2D = None
                    else:
                        mask_log_2D = None


                    all_spots, threshold = detection_without_segmentation(
                                rna=rna_img,
                                sigma=sigma,
                                min_distance=min_distance,
                                threshold = threshold,
                                mask_log_2D = mask_log_2D,
                        min_cos_tetha=min_cos_tetha,
                        order=order,
                        remove_non_sym=remove_non_sym,
                    )

                dico_spots[path_round.name][path_rna_img.name] = all_spots
                #np.save("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/try.npy", dico_spots)
                pos = "pos" + path_rna_img.name.split('pos')[1].split('_')[0]
                dico_threshold[path_round.name][pos] = threshold
            except FileNotFoundError as e:
                print(e)
                print(f"no image found for {path_rna_img.name}")
                continue

    return dico_spots, dico_threshold



