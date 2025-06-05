

import tifffile
from autofish_analysis.registration import compute_euler_transform
import numpy as np
from matplotlib import pyplot as plt
import napari

from scipy.ndimage import gaussian_filter



if __name__ == "__main__":


    fixed_image = tifffile.imread("/home/tom/Bureau/2023-10-06_LUSTRA/r1/r1_pos3_ch0.tif")
    moving_image = tifffile.imread("/home/tom/Bureau/2023-10-06_LUSTRA/r6/r6_pos3_ch0.tif")

    from skimage.exposure import rescale_intensity

    image1 = np.amax(fixed_image, 0)
    pa_ch1, pb_ch1 = np.percentile(image1, (1, 99))
    image1 = rescale_intensity(image1, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
    plt.imshow(image1)
    plt.show()

    image1 = np.amax(moving_image, 0)
    pa_ch1, pb_ch1 = np.percentile(image1, (1, 99))
    image1 = rescale_intensity(image1, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
    plt.imshow(image1)
    plt.show()




    fixed_image_mip = np.amax(fixed_image, 0)
    moving_image_mip = np.max(moving_image, 0)

    fixed_image_mip = gaussian_filter(fixed_image_mip, sigma=0.8).astype(float)
    moving_image_mip = gaussian_filter(moving_image_mip, sigma=0.8).astype(float)


    final_metric_value, thetha, x_translation, y_translation = compute_euler_transform(fixed_image_mip,
                            moving_image_mip,  # works ok
                            ndim = 2,
                            numberOfHistogramBins=500,
                            sampling_percentage=0.05,
                            max_rotation_accepted = 0.02,
                            ########## ITK parameters, leave as default unless not converging  ##########
                            learningRate=1,
                            numberOfIterations=100,
                            convergenceMinimumValue=1e-6,
                            convergenceWindowSize=10,
                            shrinkFactors = [4 ,2 ,1],
                            smoothingSigmas=[2 ,1 ,0],)

    print(final_metric_value, thetha, x_translation, y_translation)



    fixed_image_mip = np.amax(fixed_image[:, 0], 0)
    moving_image_mip = np.max(moving_image[:, 1], 0)

    fixed_image_mip = gaussian_filter(fixed_image_mip, sigma=0.8).astype(float)
    moving_image_mip = gaussian_filter(moving_image_mip, sigma=0.8).astype(float)


    final_metric_value, thetha, x_translation, y_translation = compute_euler_transform(fixed_image_mip,
                            moving_image_mip,  # works ok
                            ndim = 2,
                            numberOfHistogramBins=500,
                            sampling_percentage=0.05,
                            max_rotation_accepted = 0.02,
                            ########## ITK parameters, leave as default unless not converging  ##########
                            learningRate=1,
                            numberOfIterations=100,
                            convergenceMinimumValue=1e-6,
                            convergenceWindowSize=10,
                            shrinkFactors = [4 ,2 ,1],
                            smoothingSigmas=[2 ,1 ,0],)
    dico_translation = {"thetha": thetha, "x_translation": x_translation, "y_translation": y_translation}
    plot_registrered_image(

        dico_translation,
        path_image1="/home/tom/Bureau/phd/to_delete/For-Annalysis/S3-1.tif",
        path_image2="/home/tom/Bureau/phd/to_delete/For-Annalysis/S3-2_RNA-Cy3-MMStack_6-Pos000_000.ome.tif",
        plot_napari=True,
        figsize=(10, 10))

    figsize = (20, 20)


    fixed_image_mip = np.amax(fixed_image[:, 0], 0)
    moving_image_mip = np.max(moving_image[:, 1], 0)


    from autofish_analysis.registration import  shift
    image1 = fixed_image_mip
    image2 = moving_image_mip

    shifted_image = shift(image2, translation=(x_translation, y_translation))

    pa_ch1, pb_ch1 = np.percentile(image1, (1, 99))
    image1 = rescale_intensity(image1, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)

    pa_ch2, pb_ch2 = np.percentile(image2, (1, 99))
    image2 = rescale_intensity(image2, in_range=(pa_ch2, pb_ch2), out_range=np.uint8)

    pa_ch3, pb_ch3 = np.percentile(shifted_image, (1, 99))
    shifted_image = rescale_intensity(shifted_image, in_range=(pa_ch3, pb_ch3), out_range=np.uint8)

    fig, ax = plt.subplots(figsize=figsize, ncols=2, )
    ax[0].imshow(image1, alpha=0.5, cmap="RdGy_r")
    ax[0].imshow(image2, alpha=0.5, cmap="Greens_r")
    ax[1].imshow(image1, alpha=0.5, cmap="RdGy_r")
    ax[1].imshow(shifted_image, alpha=0.5, cmap="Greens_r")
    plt.show()


    fixed_image_mip = np.amax(fixed_image[:, 0], 0)
    moving_image_mip = np.max(moving_image[:, 1], 0)

    moving_image_mip_new = np.zeros_like(moving_image_mip)

    #moving_image_mip_new[: -round(y_translation), : -round(x_translation)] =  moving_image_mip[round(y_translation):, round(x_translation):] ## if positive
    moving_image_mip_new[-round(y_translation):, -round(x_translation):] = moving_image_mip[:round(y_translation),  :round(x_translation)] ## if negative

    pa_ch1, pb_ch1 = np.percentile(fixed_image_mip, (1, 99))
    fixed_image_mip = rescale_intensity(fixed_image_mip, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)

    pa_ch2, pb_ch2 = np.percentile(moving_image_mip, (1, 99))
    moving_image_mip = rescale_intensity(moving_image_mip, in_range=(pa_ch2, pb_ch2), out_range=np.uint8)

    pa_ch3, pb_ch3 = np.percentile(moving_image_mip_new, (1, 99))
    moving_image_mip_new = rescale_intensity(moving_image_mip_new, in_range=(pa_ch3, pb_ch3), out_range=np.uint8)


    fig, ax = plt.subplots(figsize=figsize, ncols=2, )
    ax[0].imshow(fixed_image_mip, alpha=0.5, cmap="RdGy_r")
    ax[0].imshow(moving_image_mip, alpha=0.5, cmap="Greens_r")
    ax[1].imshow(fixed_image_mip, alpha=0.5, cmap="RdGy_r")
    ax[1].imshow(moving_image_new1, alpha=0.5, cmap="Greens_r")
    plt.show()





    fixed_image = tifffile.imread("/home/tom/Bureau/phd/to_delete/For-Annalysis/S3-1.tif")
    moving_image = tifffile.imread("/home/tom/Bureau/phd/to_delete/For-Annalysis/S1-2-RNA-Cy5_MMStack_Pos0.ome.tif")
    moving_image_new = np.zeros_like(moving_image)
    #moving_image_new[:,:, :-round(y_translation), : -round(x_translation)] = moving_image[:,:, round(y_translation):, round(x_translation):] if positive
    moving_image_new[:,:,  -round(y_translation):, -round(x_translation):] = moving_image[:,:,:round(y_translation), :round(x_translation)]


    moving_image_new0= moving_image_new[:,0]
    moving_image_new1= moving_image_new[:,1]


    tifffile.imwrite("/home/tom/Bureau/phd/to_delete/For-Annalysis/S1-2-RNA-Cy5_MMStack_Pos0_registered.tif", moving_image_new)

    tifffile.imwrite("/home/tom/Bureau/phd/to_delete/For-Annalysis/S1-2-RNA-Cy5_MMStack_Pos0_ch0.ome.tif", moving_image_new0)
    tifffile.imwrite("/home/tom/Bureau/phd/to_delete/For-Annalysis/S1-2-RNA-Cy5_MMStack_Pos0_ch1.ome.tif", moving_image_new1)


################### nd to tiff




import tifffile
from autofish_analysis.registration import compute_euler_transform
import numpy as np
from matplotlib import pyplot as plt
import napari

from scipy.ndimage import gaussian_filter



if __name__ == "__main__":


    fixed_image = tifffile.imread("/home/tom/Bureau/2023-10-06_LUSTRA/r1/r1_pos3_ch0.tif")
    moving_image = tifffile.imread("/home/tom/Bureau/2023-10-06_LUSTRA/r2/r2_pos3_ch0.tif")


    from skimage.exposure import rescale_intensity

    image1 = np.amax(fixed_image, 0)
    pa_ch1, pb_ch1 = np.percentile(image1, (1, 99))
    image1 = rescale_intensity(image1, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
    plt.imshow(image1)
    plt.show()

    image2 = np.amax(moving_image, 0)
    pa_ch1, pb_ch1 = np.percentile(image2, (1, 99))
    image2 = rescale_intensity(image2, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
    plt.imshow(image2)
    plt.show()

    from autofish_analysis.registration import plot_registrered_image

    dict_translation = np.load("/home/tom/Bureau/2023-10-06_LUSTRA/26_oct_dico_translation.npy", allow_pickle = True).item()

    ### MODIFY the path to your
    folder_of_rounds = "/home/tom/Bureau/2023-10-06_LUSTRA/"
    path_image1 = folder_of_rounds + "r1/r1_pos3_ch0.tif"
    path_image2 = folder_of_rounds + "r2/r2_pos3_ch0.tif"

    plot_registrered_image(
        dict_translation,
        path_image1=path_image1,
        path_image2=path_image2,
        plot_napari=False)

