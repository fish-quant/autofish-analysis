

import numpy as np
import pandas

### create a dataframe with the information of the spots

dico_signal_quality = np.load("/media/tom/Transcend/lustr2023/images/23mai_dico_signal_quality0.npy",
                              allow_pickle=True).item()

all_mean_intensity = []
all_mean_snr = []
all_mean_symmetry_coef = []
all_mean_background = []

all_median_intensity = []
all_median_snr = []
all_median_symmetry_coef = []
all_median_background = []

for round_str in dico_signal_quality:
    mean_intensity = []
    mean_snr = []
    mean_symmetry_coef = []
    mean_background = []
    for image in dico_signal_quality[round_str]:
        mean_intensity += dico_signal_quality[round_str][image]["intensity"]
        mean_background += dico_signal_quality[round_str][image]["background"]
        mean_snr += dico_signal_quality[round_str][image]["snr"]
        mean_symmetry_coef += dico_signal_quality[round_str][image]["symmetry_coef"]

    print(f"round {round_str} mean intensity {np.mean(mean_intensity)}")
    print(f"round {round_str} median intensity {np.median(mean_intensity)}")
    print()
    print(f"round {round_str} mean background {np.mean(mean_background)}")
    print(f"round {round_str} median background {np.median(mean_background)}")
    print()

    print(f"round {round_str} mean snr {np.mean(mean_snr)}")
    print(f"round {round_str} median snr {np.median(mean_snr)}")
    print()

    print(f"round {round_str} mean symmetry coef {np.mean(mean_symmetry_coef)}")
    print(f"round {round_str} median symmetry coef {np.median(mean_symmetry_coef)}")

    print()
    print()
    print()

    all_mean_intensity.append(np.mean(mean_intensity))
    all_mean_snr.append(np.mean(mean_snr))
    all_mean_symmetry_coef.append(np.mean(mean_symmetry_coef))
    all_mean_background.append(np.mean(mean_background))

    all_median_intensity.append(np.median(mean_intensity))
    all_median_snr.append(np.median(mean_snr))
    all_median_symmetry_coef.append(np.median(mean_symmetry_coef))
    all_median_background.append(np.median(mean_background))

    index  = list(dico_signal_quality.keys())


df = pandas.DataFrame({"mean_intensity":all_mean_intensity,
                       "median_intensity":all_median_intensity,
                       "mean_snr":all_mean_snr,
                          "median_snr":all_median_snr,
                       "mean_symmetry_coef":all_mean_symmetry_coef,
                            "median_symmetry_coef":all_median_symmetry_coef,
                       "mean_background":all_mean_background,
                       "median_background":all_median_background
                       }, index=index)


dico_translation = np.load("/media/tom/Transcend/lustr2023/images/23mai_dico_translation_old.npy",
                            allow_pickle=True).item()
dico_translation_new = {}
for k in dico_translation:
    position = "pos" + k.split("pos")[1].split("_")[0]
    dico_translation_new[position] = dico_translation[k]

np.save("/media/tom/Transcend/lustr2023/images/23mai_dico_translation.npy",
        dico_translation_new)

##### fuse dico loacal detection


from pathlib import Path

import numpy as np
import pandas

path_folder_dico = Path("/media/tom/Transcend/lustr2023/images/folder_detection_each_round")

mai_dico_spots_local_detection1 = {}

for path_d in path_folder_dico.glob("*.npy"):

    dico = np.load(path_d, allow_pickle=True).item()

    for round in dico:
        print(round)

        mai_dico_spots_local_detection1[round] = dico[round]

np.save("/media/tom/Transcend/lustr2023/images/mai_dico_spots_local_detection1.npy",mai_dico_spots_local_detection1)


X = np.array(dico_spots['r1_Cy3']['r1_pos0_ch0.tif'])

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points_to_keep)

distances, indices = nbrs.kneighbors(points_to_keep)

import itertools

threshold = 0.3
scale_z_xy = np.array([0.3, 0.1, 0.1])
input_array = X
unique_tuple = [tuple(s) for s in input_array]
unique_tuple = list(set((unique_tuple)))

combos = itertools.combinations(unique_tuple, 2)
points_to_remove = [list(point2)
                    for point1, point2 in combos
                    if np.linalg.norm(point1 * scale_z_xy - point2 * scale_z_xy) < threshold]

points_to_keep = [point for point in unique_tuple if list(point) not in points_to_remove]



list_img = Path("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/k_r1_DAPI").glob("*")

for img_path in list_img:


    print(img_path)
    new_mane = str(img_path).replace("DAPI__", "")
    img_path.rename(new_mane)




###############

import napari
import numpy as np
import tifffile
from registration import shift

image1 = tifffile.imread("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/r1/r1_pos9_ch0.tif")
image2 = tifffile.imread("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/r2/r2_pos9_ch0.tif")

image1 = np.amax(image1, 0)
image2 = np.amax(image2, 0)

dico_translation = np.load(f"/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_translation.npy",
                           allow_pickle=True).item()

x_translation = dico_translation['pos9']['r1']['r2']['x_translation']
y_translation = dico_translation['pos9']['r1']['r2']['y_translation']

from registration import shift

shifted_image4 = shift(image2, translation=(x_translation, y_translation))

viewer = napari.view_image(image1, name="image1")
viewer.add_image(image2, name="image2")

viewer.add_image(shifted_image4, name="image_2_registered")
viewer.add_image(matched-reference, name="image_2_registered")
########### try histogram matching

import bigfish.detection as detection
import bigfish.stack as stack
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.exposure import match_histograms

reference =  stack.log_filter(shifted_image4, 1.35)# shifted_image4
image =  stack.log_filter(image1, 1.35)# image1

matched = match_histograms(image, reference, channel_axis=-1)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2, ax3, ax4):
    aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')
ax4.imshow(matched-reference)
ax4.set_title('matched-reference')
plt.tight_layout()
plt.show()

mask = matched.astype(int)-reference.astype(int)
mask[mask<0] = 0
viewer.add_image(mask, name="mask3")


mask = mask.astype(np.uint16)


rna_log = stack.log_filter(mask, 2.5)
mask_max= detection.local_maximum_detection(rna_log, min_distance=[3,3])


threshold = detection.automated_threshold_setting(rna_log, mask_max)
spots_man, _ = detection.spots_thresholding(rna_log, mask_max, threshold)


viewer.add_points(spots_man, name='spots_man2',
                  face_color='red', edge_color='red', size=5)





mask = reference.astype(int) - matched.astype(int)
mask[mask<0] = 0
viewer.add_image(mask, name="mask3")


mask = mask.astype(np.uint16)


rna_log = stack.log_filter(mask, 1.35)
mask_max= detection.local_maximum_detection(rna_log, min_distance=[3,3])


threshold = detection.automated_threshold_setting(rna_log, mask_max)
spots_man, _ = detection.spots_thresholding(rna_log, mask_max, threshold)
viewer.add_points(spots_man, name='spots_man',
                  face_color='blue', edge_color='blue', size=5)






r5_pos2_ch0.tif
###################



channel_nuc = 'ch3'
channel_bead = 'ch2'
channel_rna = 'ch0'

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage import data

path_acquisition = "/media/tom/T7/2022-02-24_opool-1/Stitch/acquisition/"
path_to_save = "/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/"
for path_round in Path(path_acquisition).glob("r*"):


    print(path_round.name)
    Path(path_to_save + path_round.name).mkdir(parents=True, exist_ok=True)
    for path_image in Path(path_round).glob("opool*ch1*.tif"):

        print(path_image.name)
        position = path_image.name.split("Pos_")[1].split("_")[0]
        channel = path_image.name.split("ch")[1].split(".")[0]

        if int(position) >= 10:
            continue
        img = tifffile.imread(path_image)

        #tifffile.imsave(path_to_save + path_round.name + "/" + "r1_pos" + str(position) +  ".tif",
        #                img)



        tifffile.imsave(path_to_save + path_round.name + "/" + path_round.name +"_" + "pos" + str(position) + "_ch" + channel + ".tif",
                        img)
        print(path_to_save + path_round.name + "/" + path_round.name +"_" + "pos" + str(position) + "_ch" + channel + ".tif")






#%%



from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage import data

dapi = tifffile.imread("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi_mip.tif")
tifffile.imsave("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi_mip.tif", dapi[ :5833, :5819])



dapi = tifffile.imread("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi.npy")
tifffile.imsave("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi.npy", dapi[ :5833, :5819])



final_mask = np.load("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_masks.npy")
np.save("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_masks.npy", final_mask[ :5833, :5819])


final_mask = np.load("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi.npy")
np.save("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_dapi.npy", final_mask[ :5833, :5819])




import napari

viewer = napari.view_image(dapi[ :5833, :5819], name="dapi")
dapi = tifffile.imread("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/3jully_final_masks_mip.tif")


viewer = napari.view_image(dapi[ :5833, :5819], name="dapi")


dict_spots_registered_df = np.load("/media/tom/Transcend/autofish_test_stiching/dict_spots_registered_df_r1_with_cell.npy", allow_pickle=True).item()



dict_stitch_img = np.load("/media/tom/Transcend/autofish_test_stiching/dict_stitch_img.npy", allow_pickle=True).item()


dico_bc_gene={
        'r1': "gene1",
        'r3': "gene2",
        }

## image dimention parameter
image_shape=[38, 2048, 2048]
nb_tiles_x=3
nb_tiles_y=1

#########

dict_spots_registered_df = np.load("/media/tom/Transcend/autofish_test_stiching/dict_spots_registered_df_r1_with_cell.npy",
                                   allow_pickle=True).item()

dict_stitch_img = np.load("/media/tom/Transcend/autofish_test_stiching/dict_stitch_img.npy",
                            allow_pickle=True).item()

dict_bc_gene={
        'r1': "gene1",
        'r3': "gene2",
        }

## image dimention parameter
image_shape=[38, 2048, 2048]
nb_tiles_x=3
nb_tiles_y=1

df_matching_new_cell_label = np.load("/media/tom/Transcend/autofish_test_stiching/df_matching_new_cell_label.npy",
                                        allow_pickle=True).item()


###############""





older_of_rounds = "/media/tom/Transcend/autofish_test_stiching/"



dict_round_gene={
        'r1': "gene1",
        'r3': "gene2",
        }

dico_spots_registered_df = np.load(Path(folder_of_rounds) /   "dico_spots_registered_stitch_df.npy",
                                   allow_pickle=True).item()
cell_column_name = "cell_assignment"
dico_spots_registered_df['img0']












