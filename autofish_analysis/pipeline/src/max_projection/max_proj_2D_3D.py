import numpy as np
from pathlib import Path
from skimage import io
from skimage.exposure import equalize_hist
from scipy.ndimage import uniform_filter
import napari
from tqdm import tqdm

from apifish.stack import get_in_focus_indices, compute_focus

from src.utils.file_handling import FileProcessor
from src.utils.parameters_tracking import Parameter_tracking as Track
from src.utils.interaction import UserInteraction as Ui
from src.utils.plots import Plots
from src.utils.interaction import UserInteraction as Ui


class MaxProj:
    def test_max_projections(self):
        print(
            """Choose an image and test the kind of max projection
        you want to do, before running the maxprojections
        in batch.
        """
        )

        fp = FileProcessor()
        ui = Ui()

        hist_eq = ui.ask_yes_no("Perform histogram equalization ?    ")
        special_mx = ui.ask_yes_no(
            "Use special max projection for the cell (computationally intensive) ?    "
        )

        im_path = fp.select_file()

        if Path(im_path).exists() and (
            Path(im_path).suffix == ".tiff" or Path(im_path).suffix == ".tif"
        ):
            im = io.imread(im_path)
            mpi = self.max_proj(im, hs_method_max_proj=special_mx)
            mpi_2 = mpi.copy()
            if hist_eq:
                mpi_2 = equalize_hist(mpi_2)

        viewer = napari.Viewer()
        viewer.add_image(mpi_2)
        napari.run()

    def max_project_all_lnp(self, constants: dict):
        print(
            """
        We use 2D images to feed the algorithm
        that segments cells and nuclei.
        When images are 2D, we can enhance them 
        numerically or if not, just copy them to a folder.
        to a folder. 
        When images are 3D we select some planes near the best
        focus to compute the max projection, and enhance them
        numerically (if needed).
        Images have to be in a different folder to train different
        models and do not have overlap between the masks.
        When images are 3D, the maxprojection can be computationally
        expensive. """
        )

        wells = constants["MODALITIES"]
        ui = Ui()
        # As there are several wells, we save all the maxprojection in a same folder
        # independetly of the well (but we create a copy for CELL and one for NUCLEI,
        # since we need to train different models in a same file).

        train_2D_folder = (
            Path(constants["PIPELINE_FOLDER"])
            / "Analysis"
            / constants["BATCH_NAME"]
            / "train_2D"
        )

        if not train_2D_folder.exists():
            train_2D_folder.mkdir(parents=True, exist_ok=True)

        train_2D_folder_nuc = train_2D_folder / "NUCLEI"
        if not train_2D_folder_nuc.exists():
            train_2D_folder_nuc.mkdir(parents=True, exist_ok=True)

        train_2D_folder_cell = train_2D_folder / "CELL"
        if not train_2D_folder_cell.exists():
            train_2D_folder_cell.mkdir(parents=True, exist_ok=True)

        # 1) Segment the NUCLEI
        chan_n = ui.ask_channel(constants, text=" Channel to segment the nuclei.")
        # 2) Segment the CELL
        chan_cb = ui.ask_channel(constants, text=" Channel to segment the cell body.")

        # 3) Choose whether to apply histogram equalization.
        hist_eq_cb = ui.ask_yes_no(
            "Perform histogram equalization for the cell body ?    "
        )
        hist_eq_nuc = ui.ask_yes_no(
            "Perform histogram equalization for the nuclei ?    "
        )
        special_mx_nuc = ui.ask_yes_no(
            "Use special max projection for the nuclei (computationally intensive) ?    "
        )
        special_mx_cell = ui.ask_yes_no(
            "Use special max projection for the cell (computationally intensive) ?    "
        )

        constants["TRAIN_2D_FOLDER_CELL"] = str(train_2D_folder_cell)
        constants["TRAIN_2D_FOLDER_NUCLEI"] = str(train_2D_folder_nuc)
        constants["CHANNEL_NUCLEI_SEG"] = chan_n
        constants["CHANNEL_CELL_SEG"] = chan_cb
        constants["HIST_EQ_CELL"] = hist_eq_cb
        constants["HIST_EQ_NUC"] = hist_eq_nuc
        constants["HS_MAX_PROJ_NUC"] = special_mx_nuc
        constants["HS_MAX_PROJ_CELL"] = special_mx_cell

        if chan_n == chan_cb:
            for well in tqdm(wells):
                print(f"processing well {well}")
                list_fish_ims = constants[f"BATCH_{well}_{chan_n}"]
                list_well_chan_mip_n = []
                list_well_chan_mip_c = []
                for im_path in list_fish_ims:
                    im_name = "".join(Path(im_path).stem.split("_")[:-1])
                    im = io.imread(im_path)
                    mpi_cell = self.max_proj(im, hs_method_max_proj=special_mx_cell)
                    mpi_nuc = self.max_proj(im, hs_method_max_proj=special_mx_nuc)
                    if hist_eq_nuc:
                        mpi_nuc = equalize_hist(mpi_nuc)
                    if hist_eq_cb:
                        mpi_cell = equalize_hist(mpi_cell)
                    io.imsave(train_2D_folder_nuc / (im_name + ".tiff"), mpi_nuc)
                    io.imsave(train_2D_folder_cell / (im_name + ".tiff"), mpi_cell)
                    list_well_chan_mip_n.append(
                        str(train_2D_folder_nuc / (im_name + ".tiff"))
                    )
                    list_well_chan_mip_c.append(
                        str(train_2D_folder_cell / (im_name + ".tiff"))
                    )

                constants[f"BATCH_{well}_{chan_n}_NUCLEI_MIP"] = list_well_chan_mip_n
                constants[f"BATCH_{well}_{chan_cb}_CELL_MIP"] = list_well_chan_mip_c
        else:
            for well in wells:
                print(f"processing well {well}")
                # segment cells
                list_fish_ims = constants[f"BATCH_{well}_{chan_cb}"]
                list_well_chan_mip = []
                for im_path in list_fish_ims:
                    im_name = "".join(Path(im_path).stem.split("_")[:-1])
                    im = io.imread(im_path)
                    mpi = self.max_proj(im, hs_method_max_proj=special_mx_cell)
                    if hist_eq_cb:
                        mpi = equalize_hist(mpi)
                    io.imsave(train_2D_folder_cell / (im_name + ".tiff"), mpi)
                    list_well_chan_mip.append(
                        str(Path(train_2D_folder_cell) / (im_name + ".tiff"))
                    )
                constants[f"BATCH_{well}_{chan_cb}_CELL_MIP"] = list_well_chan_mip

                # segment nuclei
                list_fish_ims = constants[f"BATCH_{well}_{chan_n}"]
                list_well_chan_mip = []
                for im_path in list_fish_ims:
                    im_name = "".join(Path(im_path).stem.split("_")[:-1])
                    im = io.imread(im_path)
                    mpi = self.max_proj(im, hs_method_max_proj=special_mx_nuc)
                    if hist_eq_nuc:
                        mpi = equalize_hist(mpi)
                    io.imsave(train_2D_folder_nuc / (im_name + ".tiff"), mpi)
                    list_well_chan_mip.append(
                        str(Path(train_2D_folder_nuc) / (im_name + ".tiff"))
                    )
                constants[f"BATCH_{well}_{chan_n}_NUCLEI_MIP"] = list_well_chan_mip

        print("Max projection executed succesfully")
        tk = Track()
        tk.save_constants_and_commit_hash(
            constants,
            constants["BATCH_NAME"],
            folder_path=constants["JSON_FILE_ADDRESS"],
        )
        print("updating the json file")

    def max_proj(self, im: np.ndarray, hs_method_max_proj=True) -> np.ndarray:
        """On 2d images, just make a copy. On 3D images compute
        the maxprojection on the zstacks with a higher focus
        using Helmli and Scherer’s method."""

        if im.ndim == 3:
            if hs_method_max_proj:
                NEIGHBORHOOD_SIZE_FOCUS = 31  # parameter used for focus computation
                PROPORTION_FOCUS = int(5 * 100 / np.shape(im)[0])
                focus = self.compute_focus_opt(
                    im, neighborhood_size=NEIGHBORHOOD_SIZE_FOCUS
                )
                inds = get_in_focus_indices(focus, PROPORTION_FOCUS)
                im_sub = im[np.array(inds, dtype=int), :, :]
                mip_im = np.max(im_sub, axis=0)
            else:
                mip_im = np.max(im, axis=0)
        elif im.ndim == 2:
            mip_im = im.copy()
        return mip_im

    def observe_max_projection_results(self, constants: dict):
        # pts = Plots()
        ui = Ui()

        well  = ui.ask_list(constants, 'MODALITIES')
        struc = ui.ask_list(constants, "STRUCTURES", text="structure")

        chan_dap_fish = constants[f"CHANNEL_{struc}_SEG"]
        list_im = constants[f"BATCH_{well}_{chan_dap_fish}_{struc}_MIP"]

        viewer_mp_s = napari.Viewer(title=f" {well} {chan_dap_fish} {struc} MIP")
        # cm = pts.custom_color_map_for_napari(
        #     pow=0.5, nb_points=256
        # )  # non linear colormap

        for file_path in list_im:
            im = io.imread(file_path)
            viewer_mp_s.add_image(
                im,
                name=f"MIP {Path(file_path).stem}",  #  colormap=("custom_gray", cm)
            )
        napari.run()

    def compute_focus_opt(self, image: np.ndarray, neighborhood_size=31):
        """Helmli and Scherer’s mean method is used as a focus metric.
        Optimized version: replace mean_filter by uniform_filter. 16/07/25.


        For each pixel yx in a 2-d image, we compute the ratio:

        .. math::

            R(y, x) = \\left \\{ \\begin{array}{rcl} \\frac{I(y, x)}{\\mu(y, x)} &
            \\mbox{if} & I(y, x) \\ge \\mu(y, x) \\\ \\frac{\\mu(y, x)}{I(y, x)} &
            \\mbox{otherwise} & \\end{array} \\right.

        with :math:`I(y, x)` the intensity of the pixel yx and :math:`\\mu(y, x)`
        the mean intensity of the pixels in its neighborhood.

        For a 3-d image, we compute this metric for each z surface.

        Parameters
        ----------
        image : np.ndarray
            A 2-d or 3-d image with shape (y, x) or (z, y, x).
        neighborhood_size : int or tuple or list, default=31
            The size of the square used to define the neighborhood of each pixel.
            An odd value is preferred. To define a rectangular neighborhood, a
            tuple or a list with two elements (height, width) can be provided.

        Returns
        -------
        focus : np.ndarray, np.float64
            A 2-d or 3-d tensor with the R(y, x) computed for each pixel of the
            original image.

        """

        # cast image in float if necessary
        if image.dtype in [np.uint8, np.uint16, np.int32, np.int64]:
            image_float = image.astype(np.float64)
        else:
            image_float = image

        # build kernel
        if image.ndim == 3:
            focus = self._compute_focus_3d_opt(image_float, neighborhood_size)
        else:
            focus = self._compute_focus_2d_opt(image_float, neighborhood_size)

        return focus

    def _compute_focus_3d_opt(self, image_3d, kernel_size):
        """Compute a pixel-wise focus metric for a 3-d image.

        Parameters
        ----------
        image_3d : np.ndarray, np.float
            A 3-d image with shape (z, y, x).
        kernel_size : int or tuple or list
            The size of the square used to define a kernel size. An odd value is
            preferred. To define a rectangular kernel, a tuple or a list with two
            elements (height, width) can be provided.

        Returns
        -------
        focus : np.ndarray, np.float
            A 3-d tensor with the R(z, y, x) computed for each pixel of the
            original image.

        """
        # compute focus metric for each z surface
        focus = []
        for image_2d in image_3d:
            focus_2d = self._compute_focus_2d_opt(image_2d, kernel_size)
            focus.append(focus_2d)

        #  stack focus metrics
        focus = np.stack(focus).astype(image_3d.dtype)

        return focus

    def _compute_focus_2d_opt(self, image_2d, kernel_size):
        """Compute a pixel-wise focus metric for a 2-d image.

        Parameters
        ----------
        image_2d : np.ndarray, np.float
            A 2-d image with shape (y, x).
        kernel_size : int or tuple or list
            The size of the square used to define a kernel size. An odd value is
            preferred. To define a rectangular kernel, a tuple or a list with two
            elements (height, width) can be provided.

        Returns
        -------
        focus : np.ndarray, np.float
            A 2-d tensor with the R(y, x) computed for each pixel of the original
            image.

        """

        if not np.issubdtype(image_2d.dtype, np.floating):
            image_2d = image_2d.astype(np.float64)

        if isinstance(kernel_size, int):
            filter_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            filter_size = tuple(kernel_size)
        else:
            raise ValueError("kernel_size must be an int or a tuple/list of two ints.")

        image_filtered_mean = uniform_filter(image_2d, size=filter_size, mode="reflect")

        # compute focus metric
        ratio_default_1 = np.ones_like(image_2d)
        ratio_default_2 = np.ones_like(image_filtered_mean)
        ratio_1 = np.divide(
            image_2d,
            image_filtered_mean,
            out=ratio_default_1,
            where=image_filtered_mean > 0,
        )
        ratio_2 = np.divide(
            image_filtered_mean, image_2d, out=ratio_default_2, where=image_2d > 0
        )
        focus = np.where(image_2d >= image_filtered_mean, ratio_1, ratio_2)

        # cast focus dtype (np.float32 or np.float64)
        focus = focus.astype(image_2d.dtype)

        return focus
