import numpy as np
import pandas as pd
import apifish.stack as stack
from apifish.detection import get_object_radius_pixel, get_spot_volume, get_spot_surface
from segmentation.refine_seg import Segmentation
from apifish.classification import prepare_extracted_data, features_dispersion


class Synthesis:
     
    def intensity_ditributions(self, rna: np.ndarray, dic_local_var: dict):
        sg = Segmentation()
        list_df_raw = []  # list of dataframes with the intensity distributions, to be contained at the end of the function
        list_df_dv = []

        if "spots_bf_raw" in dic_local_var.keys():
            list_df_raw.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_bf_raw"], rna),
                    columns=["I_Bigfish_raw"],
                )
            )
        if "spots_uf_raw" in dic_local_var.keys():
            list_df_raw.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_uf_raw"], rna),
                    columns=["I_Ufish_raw"],
                )
            )
        if "spots_dw_raw" in dic_local_var.keys():
            list_df_raw.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_dw_raw"], rna),
                    columns=["I_Deconwolf_raw"],
                )
            )
        if "spots_sm_raw" in dic_local_var.keys():
            list_df_raw.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_sm_raw"], rna),
                    columns=["I_SpotMax_raw"],
                )
            )
        if "spots_sf_raw" in dic_local_var.keys():
            list_df_raw.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_sf_raw"], rna),
                    columns=["I_Spotiflow_raw"],
                )
            )

        if "spots_bf_dv" in dic_local_var.keys():
            list_df_dv.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_bf_dv"], rna),
                    columns=["I_Bigfish_dv"],
                )
            )
        if "spots_uf_dv" in dic_local_var.keys():
            list_df_dv.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_uf_dv"], rna),
                    columns=["I_Ufish_dv"],
                )
            )
        if "spots_dw_dv" in dic_local_var.keys():
            list_df_dv.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_dw_dv"], rna),
                    columns=["I_Deconwolf_dv"],
                )
            )
        if "spots_sm_dv" in dic_local_var.keys():
            list_df_dv.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_sm_dv"], rna),
                    columns=["I_SpotMax_dv"],
                )
            )
        if "spots_sf_dv" in dic_local_var.keys():
            list_df_dv.append(
                pd.DataFrame(
                    data=sg.extract_intensities(dic_local_var["spots_sf_dv"], rna),
                    columns=["I_Spotiflow_dv"],
                )
            )

        return list_df_raw, list_df_dv

    def compute_snr_spots_v2(self, image, spots, voxel_size, spot_radius):
        """Compute signal-to-noise ratio (SNR) based on spot coordinates.

        .. math::

            \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
            \\mbox{mean(background)}}{\\mbox{std(background)}}

        Background is a region twice larger surrounding the spot region. Only the
        y and x dimensions are taking into account to compute the SNR.

        Parameters
        ----------
        image : np.ndarray
            Image with shape (z, y, x) or (y, x).
        spots : np.ndarray
            Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
            One coordinate per dimension (zyx or yx coordinates).
        voxel_size : int, float, Tuple(int, float), List(int, float) or None
            Size of a voxel, in nanometer. One value per spatial dimension (zyx or
            yx dimensions). If it's a scalar, the same value is applied to every
            dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
            provided.
        spot_radius : int, float, Tuple(int, float), List(int, float) or None
            Radius of the spot, in nanometer. One value per spatial dimension (zyx
            or yx dimensions). If it's a scalar, the same radius is applied to
            every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
            are provided.

        Returns
        -------
        snr : float
            Median signal-to-noise ratio computed for every spots.

        std_snr: float
            Std of the distribution of SNR.

        V2: modified version, in order to estimate also the variance.

        """
        # check parameters
        stack.check_array(
            image, ndim=[2, 3], dtype=[np.uint8, np.uint16, np.float32, np.float64]
        )
        stack.check_range_value(image, min_=0)
        stack.check_array(
            spots, ndim=2, dtype=[np.float32, np.float64, np.int32, np.int64]
        )
        stack.check_parameter(
            voxel_size=(int, float, tuple, list), spot_radius=(int, float, tuple, list)
        )

        # check consistency between parameters
        ndim = image.ndim
        if ndim != spots.shape[1]:
            raise ValueError(
                "Provided image has {0} dimensions but spots are "
                "detected in {1} dimensions.".format(ndim, spots.shape[1])
            )
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError(
                    "'voxel_size' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim)
                )
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError(
                    "'spot_radius' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim)
                )
        else:
            spot_radius = (spot_radius,) * ndim

        # cast spots coordinates if needed
        if spots.dtype == np.float64:
            spots = np.round(spots).astype(np.int64)

        # cast image if needed
        image_to_process = image.copy().astype(np.float64)

        # clip coordinate if needed
        if ndim == 3:
            spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
            spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
            spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
        else:
            spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
            spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

        # compute radius used to crop spot image
        radius_pixel = get_object_radius_pixel(
            voxel_size_nm=voxel_size, object_radius_nm=spot_radius, ndim=ndim
        )
        radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
        radius_signal_ = tuple(radius_signal_)

        # compute the neighbourhood radius
        radius_background_ = tuple(i * 2 for i in radius_signal_)

        # ceil radii
        radius_signal = np.ceil(radius_signal_).astype(int)
        radius_background = np.ceil(radius_background_).astype(int)

        # loop over spots
        snr_spots = []
        background_spots = []
        for spot in spots:
            # extract spot images
            spot_y = spot[ndim - 2]
            spot_x = spot[ndim - 1]
            radius_signal_yx = radius_signal[-1]
            radius_background_yx = radius_background[-1]
            edge_background_yx = radius_background_yx - radius_signal_yx
            if ndim == 3:
                spot_z = spot[0]
                radius_background_z = radius_background[0]
                max_signal = image_to_process[spot_z, spot_y, spot_x]
                spot_background_, _ = get_spot_volume(
                    image_to_process,
                    spot_z,
                    spot_y,
                    spot_x,
                    radius_background_z,
                    radius_background_yx,
                )
                spot_background = spot_background_.copy()

                # discard spot if cropped at the border (along y and x dimensions)
                expected_size = (2 * radius_background_yx + 1) ** 2
                actual_size = spot_background.shape[1] * spot_background.shape[2]
                if expected_size != actual_size:
                    continue

                # remove signal from background crop
                spot_background[
                    :,
                    edge_background_yx:-edge_background_yx,
                    edge_background_yx:-edge_background_yx,
                ] = -1
                spot_background = spot_background[spot_background >= 0]

            else:
                max_signal = image_to_process[spot_y, spot_x]
                spot_background_, _ = get_spot_surface(
                    image_to_process, spot_y, spot_x, radius_background_yx
                )
                spot_background = spot_background_.copy()

                # discard spot if cropped at the border
                expected_size = (2 * radius_background_yx + 1) ** 2
                if expected_size != spot_background.size:
                    continue

                # remove signal from background crop
                spot_background[
                    edge_background_yx:-edge_background_yx,
                    edge_background_yx:-edge_background_yx,
                ] = -1
                spot_background = spot_background[spot_background >= 0]

            # compute mean background
            mean_background = np.mean(spot_background)
            background_spots.append(mean_background)

            # compute standard deviation background
            std_background = np.std(spot_background)

            # compute SNR
            snr = (max_signal - mean_background) / std_background
            snr_spots.append(snr)

        #  average SNR
        if len(snr_spots) == 0:
            snr_out = 0.0
            std_snr = 0
            back_med = 0
            back_std = 0
        else:
            snr_out = np.median(snr_spots)
            std_snr = np.std(snr_spots)
            back_med = np.median(background_spots)
            back_std = np.std(background_spots)

        return snr_out, std_snr, back_med, back_std

    def compute_snr_df(self, image: np.ndarray, df: pd.DataFrame, voxel_size, spot_radius) -> pd.DataFrame:
        df_snr_back = df.copy()
        
        if len(df_snr_back):
            df_snr_back['snr'] = np.nan
            df_snr_back['background'] = np.nan

            # compute radius used to crop spot image once
            radius_pixel = get_object_radius_pixel(
                voxel_size_nm=voxel_size, object_radius_nm=spot_radius, ndim=image.ndim
            )

            def compute_and_assign(row):
                if not row['in_mask']:
                    return pd.Series([np.nan, np.nan])
                y, x = int(row['Y']), int(row['X'])
                z = int(row['Z']) if 'Z' in row and image.ndim == 3 else None
                snr, background = self.compute_snr_back_single_spot(
                    image, y, x, radius_pixel, z=z
                )
                return pd.Series([snr, background])
            
            df_snr_back[['snr', 'background']] = df_snr_back.apply(compute_and_assign, axis=1)
        else:
            df_snr_back['snr']        = []
            df_snr_back['background'] = []
        
        return df_snr_back


    def compute_snr_back_single_spot_V0(self, image: np.ndarray, y: int, x:int, voxel_size, spot_radius, radius_pixel, z=None):
        """
        
        Adapted from compute_snr_spots (Artur Imbert, BigFish).

        Args:
            image (np.ndarray): 2D or 3D image
            y (int): vertical position
            x (int): horizontal position
            voxel_size (_type_): 
            spot_radius (_type_): 
            z (_type_, optional): Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            float: snr, background
        """
        stack.check_array(
            image, ndim=[2, 3], dtype=[np.uint8, np.uint16, np.float32, np.float64]
        )
        stack.check_range_value(image, min_=0)
        stack.check_parameter(
            voxel_size=(int, float, tuple, list), spot_radius=(int, float, tuple, list)
        )
        
        if np.isnan(y) or np.isnan(x):
            return np.nan, np.nan
        if z is not None:
            if np.isnan(z):
                return np.nan, np.nan
                
        if z is not None:
            spot = np.array([z, y, x])
        else:
            spot = np.array([y, x])     
        stack.check_array(
            spot, ndim=1, dtype=[np.float32, np.float64, np.int32, np.int64]
        )
        
        # check consistency between parameters
        ndim = image.ndim
        if ndim != spot.shape[0]:
            raise ValueError(
                "Provided image has {0} dimensions but spots are "
                "detected in {1} dimensions.".format(ndim, spot.shape[0])
            )
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError(
                    "'voxel_size' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim)
                )
        else:
            voxel_size = (voxel_size,) * ndim
        
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError(
                    "'spot_radius' must be a scalar or a sequence with {0} "
                    "elements.".format(ndim)
                )
        else:
            spot_radius = (spot_radius,) * ndim
        
        # cast spots coordinates if needed
        if spot.dtype == np.float64:
            spot = np.round(spot).astype(np.int64)

        # cast image if needed
        image_to_process = image.copy().astype(np.float64)
        
        # clip coordinate if needed
        if ndim == 3:
            spot[0] = np.clip(spot[0], 0, image_to_process.shape[0] - 1)
            spot[1] = np.clip(spot[1], 0, image_to_process.shape[1] - 1)
            spot[2] = np.clip(spot[2], 0, image_to_process.shape[2] - 1)
        else:
            spot[0] = np.clip(spot[0], 0, image_to_process.shape[0] - 1)
            spot[1] = np.clip(spot[1], 0, image_to_process.shape[1] - 1)


        radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
        radius_signal_ = tuple(radius_signal_)
        
        # compute the neighbourhood radius
        radius_background_ = tuple(i * 2 for i in radius_signal_)

        # ceil radii
        radius_signal = np.ceil(radius_signal_).astype(int)
        radius_background = np.ceil(radius_background_).astype(int)
        
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            spot_background_, _ = get_spot_volume(
                image_to_process,
                spot_z,
                spot_y,
                spot_x,
                radius_background_z,
                radius_background_yx,
            )
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border (along y and x dimensions)
            expected_size = (2 * radius_background_yx + 1) ** 2
            actual_size = spot_background.shape[1] * spot_background.shape[2]
            if expected_size != actual_size:
                return np.nan, np.nan

            # remove signal from background crop
            spot_background[
                :,
                edge_background_yx:-edge_background_yx,
                edge_background_yx:-edge_background_yx,
            ] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx
            )
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border
            expected_size = (2 * radius_background_yx + 1) ** 2
            if expected_size != spot_background.size:
                return np.nan, np.nan


            # remove signal from background crop
            spot_background[
                edge_background_yx:-edge_background_yx,
                edge_background_yx:-edge_background_yx,
            ] = -1
            spot_background = spot_background[spot_background >= 0]
    
    
        # compute mean background
        mean_background = np.mean(spot_background)

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        
        return snr, mean_background
        
 
    def compute_snr_back_single_spot(self, image: np.ndarray, y: int, x: int, radius_pixel, z=None):
        if np.isnan(y) or np.isnan(x) or (z is not None and np.isnan(z)):
            return np.nan, np.nan

        ndim = image.ndim
        spot = np.array([z, y, x] if z is not None else [y, x], dtype=np.int64)

        # Clip coordinates to image bounds
        for i in range(ndim):
            spot[i] = np.clip(spot[i], 0, image.shape[i] - 1)

        # Compute signal and background radii
        radius_signal = np.ceil([np.sqrt(ndim) * r for r in radius_pixel]).astype(int)
        radius_background = (2 * radius_signal).astype(int)

        spot_y, spot_x = spot[-2], spot[-1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx

        if ndim == 3:
            spot_z = spot[0]
            max_signal = image[spot_z, spot_y, spot_x]
            spot_background_, _ = get_spot_volume(
                image, spot_z, spot_y, spot_x, radius_background[0], radius_background_yx
            )
            if spot_background_.shape[1:] != (2 * radius_background_yx + 1, 2 * radius_background_yx + 1):
                return np.nan, np.nan

            # Cast to float64 to allow -1 assignment
            spot_background = spot_background_.astype(np.float64)
            spot_background[:, edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1

        else:
            max_signal = image[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(image, spot_y, spot_x, radius_background_yx)
            if spot_background_.shape != (2 * radius_background_yx + 1, 2 * radius_background_yx + 1):
                return np.nan, np.nan

            spot_background = spot_background_.astype(np.float64)
            spot_background[edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1

        # Filter valid background pixels
        valid_background = spot_background[spot_background >= 0]
        if valid_background.size == 0:
            return np.nan, np.nan

        mean_background = np.mean(valid_background)
        std_background  = np.std(valid_background)
        snr = (max_signal - mean_background) / std_background if std_background > 0 else np.nan

        return snr, mean_background

    def colocalisation_analysis(
        self, gene1_spot: np.ndarray, gene2_spot: np.ndarray, thresh_dist=None
    ):
        """associates together spots that are closer than a threshold.

        Args:
            gene1_spot (np.ndarray): positions of the first gene detected (of dimensions 2 or 3)
            gene2_spot (np.ndarray): positions of the second gene detected.
            thresh_dist (_type_, optional): Defaults to np.sqrt(2).


        Returns:
            gene1_alone_number (int): number of positions in which the gene 1 is alone.
            gene2_alone_number (int): number of positions in which the gene 2 is alone.
            gene_a_b_together_num (int):  number of positions in which both genes colocalize.
            positions_overlap  (np.ndarray): coordinates of overlapping points.
        """
        if thresh_dist is None:
            n = len(gene1_spot[0])
            thresh_dist = np.ceil(np.sqrt(n) * 100) / 100

        dist_mat = np.zeros((len(gene1_spot), len(gene2_spot)))
        for ind_l, coord_g1 in enumerate(gene1_spot):
            for ind_c, coord_g2 in enumerate(gene2_spot):
                dist_mat[ind_l, ind_c] = self.euclidean_dist(coord_g1, coord_g2)

        dist_mat_thresh = dist_mat.copy()
        dist_mat_thresh[dist_mat > thresh_dist] = np.nan

        dist_mat_thresh_c = dist_mat_thresh.copy()
        gene1_spot_c = gene1_spot.copy()
        gene2_spot_c = gene2_spot.copy()

        list_gene1_only = []
        list_gene2_only = []
        list_gene1_gene2 = []

        while len(dist_mat_thresh_c):
            coord_start = gene1_spot_c[0, :]
            line_dist = dist_mat_thresh_c[0]
            ind_2 = self.nanargmin(line_dist)
            if isinstance(ind_2, np.ndarray):  # proxi for np.array([])
                # remove first element from list 1
                list_gene1_only.append(coord_start)
                gene1_spot_c = np.delete(gene1_spot_c, 0, axis=0)
                gene1_spot_c = np.atleast_2d(gene1_spot_c)
                dist_mat_thresh_c = np.delete(dist_mat_thresh_c, 0, axis=0)
                dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
            else:
                col_dist = dist_mat_thresh_c[:, ind_2]
                ind_1 = self.nanargmin(col_dist)
                if isinstance(ind_1, np.ndarray):
                    # retirer b
                    list_gene2_only.append(gene2_spot_c[ind_2, :])
                    gene2_spot_c = np.delete(gene2_spot_c, ind_2, axis=0)
                    gene2_spot_c = np.atleast_2d(gene2_spot_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_2, axis=1)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
                else:
                    # retirer b, a'
                    list_gene1_gene2.append(
                        (gene1_spot_c[ind_1, :], gene2_spot_c[ind_2, :])
                    )
                    gene1_spot_c = np.delete(gene1_spot_c, ind_1, axis=0)
                    gene1_spot_c = np.atleast_2d(gene1_spot_c)
                    gene2_spot_c = np.delete(gene2_spot_c, ind_2, axis=0)
                    gene2_spot_c = np.atleast_2d(gene2_spot_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_2, axis=1)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_1, axis=0)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)

        if len(gene2_spot_c):
            for coords in gene2_spot_c:
                list_gene2_only.append(coords)

        return list_gene1_only, list_gene2_only, list_gene1_gene2

    def euclidean_dist(self, x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.sum((x - y) ** 2))

    def nanargmin(self, arr, axis=None, nan_position="last"):
        """
        Find the indices of the minimum values along an axis ignoring NaNs.
        Similar to np.nanargmin, but handles edge cases and nan_position more robustly.

        Args:
            arr (numpy.ndarray): The input array.
            axis (int, optional): The axis along which to operate. Defaults to None,
                                meaning find the minimum of the flattened array.
            nan_position (str, optional): Where to put NaNs in the sorted indices.
                                        'first': NaNs at the beginning.
                                        'last': NaNs at the end.
                                        Defaults to 'last'.

        Returns:
            numpy.ndarray: An array of indices of the minimum values.  Returns an
                        empty array if all elements are NaN or the input is empty.
                        Returns a scalar if axis is None.

        Raises:
            ValueError: If nan_position is not 'first' or 'last'.
        """

        if nan_position not in ("first", "last"):
            raise ValueError("nan_position must be 'first' or 'last'")

        arr = np.asanyarray(arr)  # handle potential non-numpy inputs

        if arr.size == 0:  # Handle empty array case
            return np.array(
                []
            )  # Return empty array, consistent with np.nanargmin behavior

        mask = np.isnan(arr)

        if np.all(mask):  # Handle all-NaN case
            if axis is None:
                return np.array([])  # Return empty array for consistency
            else:
                if nan_position == "first":
                    return np.zeros(
                        arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int
                    )  # Return array of zeros
                else:  # nan_position == 'last'
                    return np.full(
                        arr.shape[:axis] + arr.shape[axis + 1 :],
                        arr.shape[axis] - 1,
                        dtype=int,
                    )  # Return array of last index values

        if axis is None:
            # Flatten and handle 1D array
            flat_arr = arr.flatten()
            flat_mask = mask.flatten()
            valid_indices = np.where(~flat_mask)[0]  # Indices of non-NaN values
            if valid_indices.size == 0:  # all NaN
                return np.array([])
            min_index_flat = valid_indices[np.argmin(flat_arr[valid_indices])]
            return min_index_flat

        else:
            # Handle multi-dimensional array and specified axis
            valid_indices = np.where(~mask)

            if valid_indices[0].size == 0:  # all NaN along the axis
                if nan_position == "first":
                    return np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int)
                else:  # nan_position == 'last'
                    return np.full(
                        arr.shape[:axis] + arr.shape[axis + 1 :],
                        arr.shape[axis] - 1,
                        dtype=int,
                    )

            min_indices = np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int)

            for idx in np.ndindex(*arr.shape[:axis], *arr.shape[axis + 1 :]):
                sl = list(idx[:axis]) + [slice(None)] + list(idx[axis:])
                arr_slice = arr[tuple(sl)]
                mask_slice = mask[tuple(sl)]
                valid_slice_indices = np.where(~mask_slice)[
                    0
                ]  # Indices of non-NaN values
                if valid_slice_indices.size > 0:
                    min_index_slice = valid_slice_indices[
                        np.argmin(arr_slice[valid_slice_indices])
                    ]
                    min_indices[idx] = min_index_slice

            return min_indices

    def bin_colocalization(self, cell_masks: np.ndarray, spot_coordinates_g1: np.ndarray, spot_coordinates_g2: np.ndarray) -> pd.DataFrame:
        """
        Determines binary colocalization (presence of both gene spots) within each cell.

        Args:
            cell_masks (np.ndarray): A 2D or 3D numpy array where each element
                                    contains the integer ID of the cell it belongs to
                                    (0 for background).
            spot_coordinates_g1 (np.ndarray): A numpy array of shape (N, D) where N is
                                            the number of spots and D is the number
                                            of dimensions (e.g., 2 for 2D, 3 for 3D).
                                            Coordinates should be in pixel/voxel space.
            spot_coordinates_g2 (np.ndarray): A numpy array with the same shape as
                                            spot_coordinates_g1 for the second gene.

        Returns:
            pd.DataFrame: A DataFrame with colums, cell IDs and a
                        boolean column 'bin_coloc' indicating True if both
                        genes are present in the cell, False otherwise.
        """
 
        labs_g1 = []
        for pos in spot_coordinates_g1:
            labs_g1.append(cell_masks[pos[0], pos[1]])

        labs_g2 = []
        for pos in spot_coordinates_g2:
            labs_g2.append(cell_masks[pos[0], pos[1]])

        cell_ids_g1 =  np.array(labs_g1)
        cell_ids_g2 =  np.array(labs_g2)

        cells_with_g1 = set(cell_ids_g1)
        cells_with_g2 = set(cell_ids_g2)

        all_cell_ids = np.unique(cell_masks)
        all_cell_ids = all_cell_ids[all_cell_ids != 0]

        coloc_status = {}
        for cell_id in all_cell_ids:
                coloc_status[cell_id] = (cell_id in cells_with_g1) and (cell_id in cells_with_g2)

        df = pd.DataFrame.from_dict(coloc_status, orient='index', columns=['bin_coloc'])
        df.index.name = 'cell_id'
        df = df.reset_index()
        return df

    def spatial_statistics(self, mask_cell: np.ndarray, mask_nuc: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame, image_2D: np.ndarray):
        """Compute spatial statistics (index_polarization, index_dispersion, index_peripheral_distribution) for each cell with spots on it """
        
        df_stat_cells_ext = df_stat_cells.copy()
        if len(df_stat_cells_ext):
            ip_list  = []
            id_list  = []
            ipd_list = []

            df_spots_temp = df_spots[df_spots["in_mask"]==True][['cell_mask_num','Y','X']]

            for _, row_cell_stat in df_stat_cells_ext.iterrows():
                ip_ind, id_ind, ipd_ind = self.compute_spatial_stats(row_cell_stat, mask_cell, mask_nuc, df_spots_temp, image_2D)
                ip_list.append(ip_ind)
                id_list.append(id_ind)
                ipd_list.append(ipd_ind)

            df_stat_cells_ext['pol_ind']      = ip_list
            df_stat_cells_ext['disp_ind']     = id_list
            df_stat_cells_ext['per_dist_ind'] = ipd_list

        else:
            df_stat_cells_ext['pol_ind'] = []
            df_stat_cells_ext['disp_ind'] = []
            df_stat_cells_ext['per_dist_ind'] = []
            
        return df_stat_cells_ext
        
    def compute_spatial_stats(self, row_cell_stat, masks_cells: np.ndarray, masks_nucs: np.ndarray, df_spots: pd.DataFrame, image_2D: np.ndarray):
        
        id_cell   = row_cell_stat['Cell_ID']
        count     = row_cell_stat['counts']
        
        if count > 0:
            
            mask_c = np.zeros_like(masks_cells)
            mask_c[masks_cells == id_cell] = 1

            mask_n = np.zeros_like(masks_cells)
            mask_n[masks_nucs == id_cell] = 1
            
            df_mask =  df_spots[df_spots['cell_mask_num'] == id_cell]
            if len(df_mask):
                spots = df_mask[['Y','X']].to_numpy()
                temp = prepare_extracted_data(mask_c, nuc_mask = mask_n,  rna_coord=spots, ndim=2)
        
                (cell_mask, 
                distance_cell,
                distance_cell_normalized,
                centroid_cell,
                distance_centroid_cell,
                nuc_mask,
                cell_mask_out_nuc,
                distance_nuc,
                distance_nuc_normalized,
                centroid_nuc,
                distance_centroid_nuc,
                rna_coord_out_nuc,
                centroid_rna,
                distance_centroid_rna,
                centroid_rna_out_nuc,
                distance_centroid_rna_out_nuc,
                distance_centrosome) = temp
                
                ndim = 2
                index_polarization, index_dispersion, index_peripheral_distribution = features_dispersion(image_2D, spots, centroid_rna,
                                                                                                          cell_mask, centroid_cell,
                                                                                                          centroid_nuc, ndim, check_input=False)
                return index_polarization, index_dispersion, index_peripheral_distribution
            else:
                raise ValueError('There should be spots in this area')
        else:
            return np.nan, np.nan, np.nan    
  
    def roi_selection_account(self, df_stats_cells: pd.DataFrame, list_cells_to_rem: np.ndarray):
        """Updates the cell stat dataframe with a column keep (booleans)"""
        df_stats_cells_ext = df_stats_cells.copy()
        df_stats_cells_ext["Keep"] = True
        mask_to_remove = df_stats_cells_ext['Cell_ID'].isin(list_cells_to_rem)
        df_stats_cells_ext.loc[mask_to_remove, 'Keep'] = False
        return df_stats_cells_ext
    
    def binary_colocalization(self, df_g1: pd.DataFrame, df_g2: pd.DataFrame, name_gene1: str, name_gene2: str, df_stats: pd.DataFrame):
        """Boolean variable whether two genes are expressed in the same cell"""
        list_cells_coloc = np.array(list(set(np.unique(df_g1[df_g1['in_cell']==1.0]['cell_mask_num'].to_numpy()).tolist())  &
                                    set(np.unique(df_g2[df_g2['in_cell']==1.0]['cell_mask_num'].to_numpy()).tolist())), dtype= int)    
        df = df_stats.copy()
        df['coloc' + '_' + name_gene1 + '_' + name_gene2] = df['Cell_ID'].isin(list_cells_coloc)*1.0
        
        df.loc[df['Keep'] == False, 'coloc' + '_' + name_gene1 + '_' + name_gene2] = np.nan
        
        return df