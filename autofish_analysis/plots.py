

import numpy as np
import tifffile
from skimage.exposure import rescale_intensity
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

path_project = "/home/tom/Bureau/2023-10-06_LUSTRA/"

dict_spots = np.load('/home/tom/Bureau/2023-10-06_LUSTRA/dict_spots_polaris.npy',
                     allow_pickle=True).item()

path_save = Path("/home/tom/Bureau/2023-10-06_LUSTRA/detection_7janv_polris/")
def plot_detected_spots(dict_spots,
                        path_project = "/home/tom/Bureau/2023-10-06_LUSTRA/",
                        figsize = (20,20),
                        color = "red",
                        spots_size = 1,
                        rescal_percentage = 99.8,
                        path_save = None,
                        ):

    for round in tqdm(dict_spots):
        if path_save is None:
            path_save = Path(path_project) / (round +"/"+ "spots_detection_fi")
        path_save.mkdir(exist_ok=True, parents=True)

        for image in tqdm(dict_spots[round]):
            img = tifffile.imread(Path(path_project) / (round + "/" +  image) )
            spots_array = dict_spots[round][image]
            if img.ndim == 3:
                img = np.amax(img, 0)
            else:
                assert img == 2


            fig, ax = plt.subplots(figsize =  figsize,ncols=2, nrows=2)
            ax[0, 0].imshow(img)
            ax[1, 0].imshow(img)

            pa_ch1, pb_ch1 = np.percentile(img, (1, rescal_percentage))
            img_rescale = rescale_intensity(img, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
            ax[0, 1].imshow(img_rescale)
            ax[1, 1].imshow(img_rescale)
            if spots_array.shape[-1] == 0:
                continue

            if spots_array.ndim == 3 and spots_array.shape[-1] == 2:
                spots_array = spots_array[0]
                if len(spots_array) == 0:
                    continue
                ax[1, 1].scatter(spots_array[:, 1], spots_array[:, 0],
                                 c=color,
                                 s=spots_size)
            else:
                ax[1, 1].scatter(spots_array[:, 2], spots_array[:, 1],
                                 c=color,
                                 s=spots_size)

            fig.savefig(path_save / Path(image).stem )

            ### napaari
            img = tifffile.imread(Path(path_project) / (round + "/" +  image) )




            viewer = napari.viewer.Viewer()

            viewer.add_image(img, name='rna')
            # viewer.add_image(img_dapi, name='rna')

            viewer.add_points(spots_array, name='spots',
                              face_color='red', edge_color='red', size=5)
            break

