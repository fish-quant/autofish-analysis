import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
from pathlib import Path

from skimage.segmentation import  expand_labels
from skimage.filters import threshold_otsu
from skimage import measure
from scipy.ndimage import binary_fill_holes, median_filter
import cellpose.models as models
from cellpose import denoise

import apifish.stack as stack

#import matplotlib.pyplot as plt    


class Segmentation:
    
    def segment_cellpose_cyto3(self, image: np.ndarray, diameter=50):
        """
        Input:
        image: np.ndarray, image containing cells.
        diameter: approximate diameter in pixels of the cells.

        return:
        mask: np.ndarray of the same size of the image. Each cell is labeled
            with a diffent integer.
        """
        model = models.Cellpose(model_type="cyto3")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            masks, _, _, _ = model.eval(image, diameter=diameter, channels=None)
        return masks

    def segment_cellpose_other_pretrained_models(
        self, image: np.ndarray, gpu=False, diameter=50, model_type="cyto2_cp3"
    ):
        """
        Input:
        image: np.ndarray, image containing cells.
        diameter: approximate diameter in pixels of the cells.
        model_type: can be 'bact_fluor_cp3', 'cyto2_cp3'
        return:
        mask: np.ndarray of the same size of the image. Each cell is labeled
            with a diffent integer.
        """
        model = models.CellposeModel(model_type=model_type)  # insterad of Cellpose
        masks, flows, styles = model.eval(image, diameter=diameter, channels=None)
        return masks

    def segment_with_custom_model(
        self, image: np.ndarray, pretrained_model_path: str, gpu=False
    ):
        """
        Input:
        image: np.ndarray, image containing cells.
        pretrained_model_path: string
        return:
        mask: np.ndarray of the same size of the image. Each cell is labeled
            with a diffent integer.
        """
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model_path)
        masks, flows, styles = model.eval(image, diameter=None)
        return masks

    def deblur_cellpose(self, image: np.ndarray, diameter=30, gpu=False):
        """
        Input:
        image: nuclei DAPI image of dimensions N.M.
        diameter: approximate diameter in pixels of the cells.
        gpu: boolean.

        Return:
        im_dn: denoised image of dimensions N.M.
        """
        dn = denoise.DenoiseModel(model_type="deblur", gpu=gpu)
        im_dn = dn.eval(image, channels=None, diameter=diameter)
        return im_dn[:, :, 0]

    def find_fish_expression_area(
        self,
        fish_stack: np.ndarray,
        masks: np.ndarray,
        smoothness: Union[int, None] = None,
    ):
        """
        Returns mask in which to look for FISH (2D).
        1 - Perform max projection on fish_stack.
        2 - Performs a simple adaptive thresholding (OTSU).
        3-  Joins both masks: thresholded mask and segmented nuclei.
        4-  Fill holes.
        5-  Smooths the edges.

        Inputs:
            fish_rescaled: 3Dimage ZxWxH.
            masks: 2D image of labeled nuclei.
        """
        fish_im = stack.maximum_projection(fish_stack)
        thresh = threshold_otsu(fish_im)
        fish_im_thresh = fish_im > thresh

        masks_bool = masks > 1
        fish_im_thresh = np.logical_or(fish_im_thresh, masks_bool)
        fish_im_thresh_filled = binary_fill_holes(fish_im_thresh)
        if smoothness:
            fish_im_thresh_filled = self.smooth_instance(
                fish_im_thresh_filled, smoothness
            )

        return fish_im_thresh_filled

    def inter_dilated_labels_and_fishexpr_area(
        self, dilated_labels, fish_im_thresh_filled
    ):
        return dilated_labels * fish_im_thresh_filled

    def dilate_labels(
        self, image: np.ndarray[int], distance: int = None
    ) -> np.ndarray[int]:
        """Expand labels in label image by distance pixels without overlapping.

        Args:
            image (np.ndarray): image of labels (int), each label stands
            for a given cell or nuclei.
            distance (int, optional): Defaults to None. In this case,
            expand the labels as much as possibe in the image.
            When distance is an int, expand the labels by distance (in pixels).

        Returns:
            np.ndarray[int]: lLabeled array, whith enlarged connected regions.
        """
        if distance is None:
            distance = np.shape(image)[0]

        return expand_labels(image, distance=distance)

    def create_mask_detection(self, rna_mip: np.ndarray, spots):
        """
        Creates a mask, in which detected fish positions are set to 1.

        rna_mip: maximal intensity projection.
        spots: np.array of n x 3, n fish detected. 3 coordinates: (number of z stack, pixel # in y, pixel # in x)
        In this configuration, the z dimension is not taken into account.
        """
        rna_mip_im = np.zeros(np.shape(rna_mip))
        for i, coordinates in enumerate(spots):
            y, x = coordinates[1], coordinates[2]
            rna_mip_im[y, x] = 1

        return rna_mip_im

    def extract_intensities_df(self, df_spots: pd.DataFrame, rna: np.ndarray):
        
        df_spots_w_intensity              = df_spots.copy()
        if len(df_spots_w_intensity):
            df_spots_w_intensity['intensity'] = np.nan
        else:
            df_spots_w_intensity['intensity'] = []
            
        if rna.ndim == 3 and 'Z' in list(df_spots_w_intensity.columns):
            df_spots_w_intensity.loc[df_spots_w_intensity['in_mask'], 'intensity'] =  \
                        df_spots_w_intensity[df_spots_w_intensity['in_mask']].apply(lambda row: rna[int(row['Z']), int(row['Y']), int(row['X'])], axis=1)
        elif  rna.ndim == 2 and 'Z' not in list(df_spots_w_intensity.columns):
            df_spots_w_intensity.loc[df_spots_w_intensity['in_mask'], 'intensity'] =  \
                        df_spots_w_intensity[df_spots_w_intensity['in_mask']].apply(lambda row: rna[int(row['Y']), int(row['X'])], axis=1)
            
        return df_spots_w_intensity    
                   
    def extract_mask_fish(self, masks: np.ndarray, distance_dilation: int = 25):
        """
        Extracts the mask of the fish expression area.
        """
        mask_fish = self.dilate_labels(masks, distance=distance_dilation)
        mask_fish = (mask_fish>0)*1
        return mask_fish


    def count_points_in_mask(self, mask: np.ndarray, points: np.ndarray):
        """
        Counts the number of points within a given mask.

        Args:
            mask: A 2D NumPy array representing the mask (1 for inside, 0 for outside).
            points: A list of tuples, where each tuple represents the (y, x) coordinates of a point.

        Returns:
            The number of points that fall within the mask.
        """
        count = 0
        for tup in points:
            if mask[tup[0], tup[1]] == 1:  # Check if the point is within the mask
                count += 1
        return count

    def count_spots_in_masks(self, masks: np.ndarray, spots: np.ndarray):
        """
        Input:
            masks: each mask is defined by an integer.
            spots: np.array of dimension n.2, where n is the number of observations.
        """
        dic_count_spots = {}

        for ind_mask, mask_num in enumerate(np.unique(masks)):
            if mask_num != 0:
                mask_temp = (masks == mask_num).astype(np.uint8)
                dic_count_spots[mask_num] = self.count_points_in_mask(mask_temp, spots)
        return dic_count_spots

    def count_spots_in_masks_df(self, masks: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame):
        """
        
        Args:
            masks (np.ndarray): np array with masks. Each integer (constant inside the mask) corresponds to a single structure. 
            df_stat_cells (pd.DataFrame): collects statistics about the cell.
            df_spots (pd.DataFrame): collect statistics about the spots.
        """
        df_stat_cells_w_counts  = df_stat_cells.copy()
        df_spots_w_mask_num     = df_spots.copy()  
               
               
        if len(df_spots_w_mask_num):       
               
            df_spots_w_mask_num['in_cell']  = 0
            df_spots_w_mask_num.loc[df_spots_w_mask_num['in_mask'] == False, 'in_cell'] = np.nan
                        
            df_stat_cells_w_counts['counts']     = np.nan
            df_spots_w_mask_num['cell_mask_num'] = np.nan
                        
            for ind_mask, row_mask in df_stat_cells_w_counts.iterrows():    
                mask_lab  = row_mask['Cell_ID']
                tot_count = 0
                for index, spot_row in df_spots_w_mask_num.iterrows():
                    if spot_row['in_mask']:
                        if masks[int(spot_row['Y']), int(spot_row['X'])] == mask_lab:
                            tot_count = tot_count + 1
                            df_spots_w_mask_num.at[index, 'in_cell']       = 1.0    
                            df_spots_w_mask_num.at[index, 'cell_mask_num'] = mask_lab

                                
                df_stat_cells_w_counts.at[ind_mask, 'counts'] = tot_count            
        else:
            df_spots_w_mask_num['in_cell']  = []
            df_stat_cells_w_counts['counts']= []
            df_spots_w_mask_num['cell_mask_num'] = []
                    
                            
        return df_stat_cells_w_counts, df_spots_w_mask_num

    def spots_in_masks(self, masks: np.ndarray, points: np.ndarray):
        """Ckecks if each point is within a ensemble of masks.
        
        masks: image (int). Each submask has a given integer 
        points: np.array of dimension n.2 , where 2 stands for (y,x).
        
        returns is_in: np.array of dimension n.2, composed of ones and zeros.
        """
        is_in         = np.zeros(len(points), dtype=int)
        mask_ints_arr = np.unique(masks)
        for point_ind, point in enumerate(points):
            for ind, u in enumerate(mask_ints_arr):
                if u != 0:
                    if masks[point[0], point[1]] == u:
                        is_in[point_ind] = 1
                        break
        return is_in        
        
        
    def spots_in_masks_v2(self, masks: np.ndarray, points: np.ndarray, mask_fish: Optional[np.ndarray] = None)-> np.ndarray:
        """
        Ckecks if each point is within a ensemble of masks.
        If a mask_fish is provided, only the points within the mask_fish are considered.
        """
        if mask_fish is not None:                    
            sel_points, _ = self.subsel_points_in_mask_fish(points, mask_fish)
            is_in = np.zeros(len(sel_points), dtype=int)
            for point_ind, point in enumerate(sel_points):
                for ind, u in enumerate(np.unique(masks)):
                    if u != 0:
                        if masks[point[0], point[1]] == u:
                            is_in[point_ind] = 1
                            break
            return is_in            
        else:
            is_in = np.zeros(len(points), dtype=int)
            for point_ind, point in enumerate(points):
                for ind, u in enumerate(np.unique(masks)):
                    if u != 0:
                        if masks[point[0], point[1]] == u:
                            is_in[point_ind] = 1
                            break
            return is_in

    def spots_in_nuclei(self, masks_nuclei: np.ndarray, spots_position:np.ndarray, spots_in_masks: np.ndarray)-> np.ndarray: # mask_clean: cells, masks_seg_clean: nuclei
        """Whetermine, among the spots in the cells, which are in the nuclei.

        Args:
            masks_nuclei (np.ndarray): one label per nuclei.
            spot_in_cell (np.ndarray): 1/0 if the spot is in the cell.
            spots_position (np.ndarray): coordinates of the spots (y,x).
            spots_in_masks (np.ndarray): 1/0 depending on whether spot is in the mask of a cell.
        Returns:
            np.ndarray: length equal to the number of spots that are equal to the spots in the cells.
                1/0 if the spot is in the nuclei.
        """        """"""
        spot_in_nuclei = []
        for u in range(len(spots_in_masks)):
            if spots_in_masks[u] == 1:
                if masks_nuclei[int(spots_position[u][0]), int(spots_position[u][1])] > 0:
                    spot_in_nuclei.append(1)
                else:
                    spot_in_nuclei.append(0)
                    
        return np.array(spot_in_nuclei)

    def spots_in_nuclei_df(self, mask_nuc: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame):
        
        df_spots_w_nuclei = df_spots.copy()
        df_stat_cells_w_nuclei_counts = df_stat_cells.copy()

        if len(df_spots_w_nuclei):
            df_spots_w_nuclei['in_nuclei'] = 0
            df_spots_w_nuclei.loc[df_spots_w_nuclei['in_mask'] == False, 'in_nuclei'] = np.nan
            df_spots_w_nuclei.loc[df_spots_w_nuclei['in_cell'] == 0, 'in_nuclei']     = np.nan
            
            df_stat_cells_w_nuclei_counts['count_nuclei'] = 0

            for index, spot_row in df_spots_w_nuclei.iterrows():
                if not pd.isna(spot_row['cell_mask_num']):
                    
                    mask  = (mask_nuc == spot_row['cell_mask_num'])
                    if mask[int(spot_row['Y']), int(spot_row['X'])]:
                        df_spots_w_nuclei.at[index, 'in_nuclei'] = 1

            df_spots_w_nuclei['in_cyto'] = df_spots_w_nuclei['in_cell'] - df_spots_w_nuclei['in_nuclei']

            for ind_mask, mask_row in df_stat_cells_w_nuclei_counts.iterrows():
                if mask_row['counts']:
                    df_stat_cells_w_nuclei_counts.at[ind_mask, 'count_nuclei'] =  df_spots_w_nuclei[df_spots_w_nuclei['cell_mask_num'] == mask_row['Cell_ID']]['in_nuclei'].sum()
                    
            df_stat_cells_w_nuclei_counts['count_cyto'] = df_stat_cells_w_nuclei_counts['counts'] - df_stat_cells_w_nuclei_counts['count_nuclei']

        else:
            df_spots_w_nuclei['in_nuclei'] = []
            
            df_stat_cells_w_nuclei_counts['count_nuclei'] = []
            df_stat_cells_w_nuclei_counts['count_cyto']   = []
            
            

        return df_stat_cells_w_nuclei_counts, df_spots_w_nuclei

    def remove_labels_from_fishmask(self, mask_fish: np.ndarray, masks_cells: np.ndarray, lab_to_rem: np.ndarray) -> np.ndarray:
        """Remove the putative cells that do not match the size and intensity criteria from the fishmask. 
        fishmask is the region to consider the gene expression. 

        Args:
            mask_fish (np.ndarray): rough region of fish expression.
            masks_cells (np.ndarray): masks of expanded segmented nuclei (putative cells).
            lab_to_rem (np.ndarray): Established list of cells to remove. Do not consider those points neither in the 
            fish expression region. 

        Returns:
            np.ndarray: cleaned fishmask.
        """
        mask_fish_cleaned = mask_fish.copy()                            
        if len(lab_to_rem):
            for lab in lab_to_rem:
                mask_fish_cleaned[masks_cells==lab] = 0        
        return mask_fish_cleaned            


    def subsel_points_in_mask_fish_V0(self, points: np.ndarray, mask_fish: np.ndarray) -> np.ndarray:
        """Chooses the points wich are in the mask_fish.

        Args:
            points (np.ndarray or dataframe): detected spots, of dimension n.3 or n.2.
            mask_fish (np.ndarray): fish mask (2d).

        Returns:
            np.ndarray: selected points  (if points is of dimension n.3, the selected points will also be n.3) .
            np.ndarray: indexes of the selected points.
        """
        if np.shape(points)[1] == 2:
            sel_points = []
            indexes    = []
            for ind, point in enumerate(points):
                if mask_fish[point[0], point[1]]:
                    sel_points.append(point)
                    indexes.append(ind)
            sel_points = np.array(sel_points)
            indexes    = np.array(indexes)
            return sel_points, indexes
        elif np.shape(points)[1] == 3:
            sel_points = []
            indexes    = []
            for ind, point in enumerate(points):
                if mask_fish[point[1], point[2]]:
                    sel_points.append(point)
                    indexes.append(ind)      
            sel_points = np.array(sel_points)
            indexes    = np.array(indexes)  
            return sel_points, indexes
   

    def subsel_points_in_mask_fish(self, df_points: pd.DataFrame, mask_fish: np.ndarray):
        condition = df_points.apply(lambda row: mask_fish.astype(bool)[int(row['Y']), int(row['X'])], axis=1)
        return df_points[condition].copy()

    def add_column_in_mask_fish(self, df_points: pd.DataFrame, mask_fish: np.ndarray):
        if len(df_points):
            condition = df_points.apply(lambda row: mask_fish.astype(bool)[int(row['Y']), int(row['X'])], axis=1)
            df_points['in_mask'] = condition.to_numpy()
        else:
            df_points['in_mask']= []
    
        return df_points
       
    def mean_intensity_in_circle(self, image, center_y, center_x, radius, z=None):
        """
        Finds the pixel mean intensity within a circle in a 2D array (image).

        Args:
            image_shape: Tuple (height, width) representing the image dimensions.
            center_y: Y-coordinate of the circle's center.
            center_x: X-coordinate of the circle's center.
            radius: Radius of the circle.

        Returns:
            Returns np.nan if the inputs are not valid.
        """
        if image.ndim == 2:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x) ** 2 + (y - center_y) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <= np.ceil(radius)
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.mean(image[y_coords, x_coords])

            else:
                return np.nan
            
        elif image.ndim == 3 and z is not None:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            depth, height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x) ** 2 + (y - center_y) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <= np.ceil(radius)
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.mean(image[z, y_coords, x_coords])
            else:
                return np.nan    
            
    def summed_intensity_in_circle(self, image, center_y, center_x, radius, z=None):
        """
        Finds the pixel mean intensity within a circle in a 2D array (image).

        Args:
            image_shape: Tuple (height, width) representing the image dimensions.
            center_y: Y-coordinate of the circle's center.
            center_x: X-coordinate of the circle's center.
            radius: Radius of the circle.

        Returns:
            Returns np.nan if the inputs are not valid.
        """
        if image.ndim == 2:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x +.5) ** 2 + (y - center_y +.5) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <= radius
                       
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.sum(image[y_coords, x_coords])

            else:
                return np.nan
            
        elif image.ndim == 3 and z is not None:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            depth, height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x + .5) ** 2 + (y - center_y + 0.5) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <=  radius
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.sum(image[z, y_coords, x_coords])
            else:
                return np.nan   
      
       
    def compute_mean_spot_intensity(
        self, image: np.ndarray, spot_location: np.ndarray, spot_sizes: np.ndarray
    ):
        """Integrates summed intensity over a circle of radius spot size, wich is 
        an estimation of the spot size.

        Args:
            image (np.ndarray): Can be 2 or 3 dimensional.
            spot_location (np.ndarray): N.2 or N.3
            spot_sizes (np.ndarray): N.1

        Returns:
            mean_intensity: np.ndarray, mean intensity. 
        """
        mean_intensity = np.zeros(len(spot_location))
        if image.ndim ==2:
            for ind, tup in enumerate(spot_location):
                y, x = tup
                radius = spot_sizes[ind]
                mean_intensity[ind] = self.mean_intensity_in_circle(image, y, x, radius)
            return mean_intensity
        elif image.ndim ==3:
            for ind, tup in enumerate(spot_location):
                z, y, x = tup
                radius = spot_sizes[ind]
                mean_intensity[ind] = self.mean_intensity_in_circle(image, y, x, radius,z=z)
            return mean_intensity        
    
    def compute_sum_spot_intensity_df(self, df: pd.DataFrame, im_rna: np.ndarray):
        
        df_sum_intensity = df.copy()
        if len(df_sum_intensity):
            df_sum_intensity['sum_intensity'] = np.nan
            
            if 'Z' in list(df.columns) and im_rna.ndim == 3:            
                df_sum_intensity.loc[df_sum_intensity['in_mask'], 'sum_intensity'] = df_sum_intensity[df_sum_intensity['in_mask']].apply(lambda row: self.summed_intensity_in_circle(im_rna,
                                                                                                                                                            int(row['Y']),
                                                                                                                                                            int(row['X']),
                                                                                                                                                            radius= row['spot_width']/2,
                                                                                                                                                            z= int(row['Z']),
                                                                                                                                                            ),
                                                                                                                                                            axis=1,
                                                                                                                        )    
            else:
                df_sum_intensity.loc[df_sum_intensity['in_mask'], 'sum_intensity'] = df_sum_intensity[df_sum_intensity['in_mask']].apply(lambda row: self.summed_intensity_in_circle(im_rna,
                                                                                                                                                            int(row['Y']),
                                                                                                                                                            int(row['X']),
                                                                                                                                                            radius= row['spot_width']/2,
                                                                                                                                                            ),
                                                                                                                                                            axis=1,
                                                                                                                        )
        else:
            df_sum_intensity['sum_intensity'] = []
            
        return df_sum_intensity
        
    def compute_mean_spot_intensity_v2(
        self, image: np.ndarray, spot_location: np.ndarray, spot_sizes: np.ndarray, mask_fish: Optional[np.ndarray] = None
    ):
        if mask_fish is None:
            mean_intensity = np.zeros(len(spot_location))
            for ind, tup in enumerate(spot_location):
                y, x = tup
                radius = spot_sizes[ind]
                mean_intensity[ind] = self.mean_intensity_in_circle(image, y, x, radius)
            return mean_intensity
        else:
            sel_points, _ = self.subsel_points_in_mask_fish(spot_location, mask_fish)
            mean_intensity = np.zeros(len(sel_points))
            for ind, tup in enumerate(sel_points):
                y, x = tup
                radius = spot_sizes[ind]
                mean_intensity[ind] = self.mean_intensity_in_circle(image, y, x, radius)
            return mean_intensity
    
    def interpolate_signal(self, signal: np.array, n=11):
        """
        Upsample signal by interpolating it.
        """
        arr = []
        for u in range(len(signal) - 1):
            if u < len(signal) - 2:
                arr.append(np.linspace(signal[u], signal[u + 1], n)[:-1])
            else:
                arr.append(np.linspace(signal[u], signal[u + 1], n))
        return np.concatenate(arr)

    def determine_spot_width(self, signal: np.array, n=11, frac= 0.75):
        """
        Determine spot width by interpolating the signal
        and taking the half width of the peak.
        """
        h = len(signal) // 2 + 1
        ind_max = np.argmax(signal[h - 1 : h + 2])
        signal_int = self.interpolate_signal(signal, n)

        if ind_max == 0:
            ind_max_c = (h - 1) * (n - 1)
        elif ind_max == 1:
            ind_max_c = h * (n - 1)
        else:
            ind_max_c = (h + 1) * (n - 1)

        #signal_int = 0.5 * (signal_int + np.flip(signal_int))
        signal_higher = (
            signal_int
            > np.min(signal_int) + (signal_int[ind_max_c] - np.min(signal_int)) * frac
        )
        ind_h = []
        for d in range(len(signal_int) // 2):
            if ind_max_c + d < len(signal_higher):
                if signal_higher[ind_max_c + d]:
                    ind_h.append(ind_max_c + d)
                else:
                    break
        for d in range(1, len(signal_int) // 2):
            if ind_max_c - d >= 0:
                if signal_higher[ind_max_c - d]:
                    ind_h.append(ind_max_c - d)
                else:
                    break

        size_spot_px = len(ind_h) / (n - 1)  # convert back in pixels
        return size_spot_px

    def determine_spot_width_using_offset(self, signal: np.array, n=11, offset = 200):
        """
        Determine spot width by interpolating the signal
        and taking the width wich is at a given distance of the offset.
        """
        h = len(signal) // 2 + 1
        ind_max = np.argmax(signal[h - 1 : h + 2])
        signal_int = self.interpolate_signal(signal, n)

        if ind_max == 0:
            ind_max_c = (h - 1) * (n - 1)
        elif ind_max == 1:
            ind_max_c = h * (n - 1)
        else:
            ind_max_c = (h + 1) * (n - 1)
       
        signal_higher = signal_int > (signal_int[ind_max_c] - offset)
        
        ind_h = []
        for d in range(len(signal_int) // 2):
            if ind_max_c + d < len(signal_higher):
                if signal_higher[ind_max_c + d]:
                    ind_h.append(ind_max_c + d)
                else:
                    break
        for d in range(1, len(signal_int) // 2):
            if ind_max_c - d >= 0:
                if signal_higher[ind_max_c - d]:
                    ind_h.append(ind_max_c - d)
                else:
                    break

        # fig, ax = plt.subplots()
        # ax.plot(signal_int)
        # ax.plot([ind_max_c, ind_max_c],[signal_int[ind_max_c], signal_int[ind_max_c]],'o')
        # ax.plot(signal_higher*300)

        size_spot_px = len(ind_h) / (n - 1)  # convert back in pixels
        return size_spot_px

    def determine_spots_width_in_window(
        self, image: np.ndarray, spots: np.ndarray, w=5, n=11, frac=0.75
    ):
        """
        Determine spot size based on the half width of the intensity
            in the window around the spot.
        Spot size: nparray of N.2 elements of the form (y, x)
        w: window size
        n: interpolation factor: number of points between pixels.
        frac: fraction of the peak.
        """
        spot_sizes = np.zeros(len(spots))
        if len(spots[0]) == 2:
            for ind, tup in enumerate(spots):
                y, x = tup
                try:
                    slice_0 = image[y - w : y + w + 1, x]
                    if slice_0.size == 0:
                        im_0 = np.nan
                    else:
                        im_0 = slice_0
                except IndexError:
                    im_0 = np.nan
                try:
                    slice_1 = image[y, x - w : x + w + 1]
                    if slice_1.size == 0:
                        im_1 = np.nan
                    else:
                        im_1 = slice_1
                except IndexError:
                    im_1 = np.nan
                try:
                    slice_2 = []
                    for u in range(w + 1):
                        slice_2.append(image[y - w + u, x - w + u])
                    for u in range(1, w + 1):
                        slice_2.append(image[y + u, x + u])
                    slice_2 = np.array(slice_2)
                    if slice_2.size == 0:
                        im_2 = np.nan
                    else:
                        im_2 = slice_2
                except IndexError:
                    im_2 = np.nan
                try:
                    slice_3 = []
                    for u in range(w + 1):
                        slice_3.append(image[y + w - u, x - w + u])
                    for u in range(1, w + 1):
                        slice_3.append(image[y - u, x + u])
                    slice_3 = np.array(slice_3)
                    if slice_3.size == 0:
                        im_3 = np.nan
                    else:
                        im_3 = slice_3                        
                except IndexError:
                    im_3 = np.nan
                if isinstance(im_0, float):
                    pw_0 = np.nan
                else:
                    pw_0 = self.determine_spot_width(im_0, n=n,frac=frac)
                if isinstance(im_1, float):
                    pw_1 = np.nan
                else:
                    pw_1 = self.determine_spot_width(im_1, n=n, frac=frac)
                if isinstance(im_2, float):
                    pw_2 = np.nan
                else:
                    pw_2 = self.determine_spot_width(im_2, n=n, frac=frac)
                if isinstance(im_3, float):
                    pw_3 = np.nan
                else:
                    pw_3 = self.determine_spot_width(im_3, n=n, frac=frac)
                std_min = np.min([pw_0, pw_1, pw_2, pw_3])
                spot_sizes[ind] = std_min

            return spot_sizes
        elif len(spots[0]) == 3:
            for ind, tup in enumerate(spots):
                z, y, x = tup
                try:
                    slice_0 = image[z, y - w : y + w + 1, x]
                    if slice_0.size == 0:
                        im_0 = np.nan
                    else:
                        im_0 = slice_0
                except IndexError:
                    im_0 = np.nan
                try:
                    slice_1 = image[z, y, x - w : x + w + 1]
                    if slice_1.size == 0:
                        im_1 = np.nan
                    else:
                        im_1 = slice_1
                except IndexError:
                    im_1 = np.nan
                try:
                    slice_2 = []
                    for u in range(w + 1):
                        slice_2.append(image[z, y - w + u, x - w + u])
                    for u in range(1, w + 1):
                        slice_2.append(image[z, y + u, x + u])
                    slice_2 = np.array(slice_2)
                    if slice_2.size == 0:
                        im_2 = np.nan
                    else:
                        im_2 = slice_2
                except IndexError:
                    im_2 = np.nan
                try:
                    slice_3 = []
                    for u in range(w + 1):
                        slice_3.append(image[z, y + w - u, x - w + u])
                    for u in range(1, w + 1):
                        slice_3.append(image[z, y - u, x + u])
                    slice_3 = np.array(slice_3)
                    if slice_3.size == 0:
                        im_3 = np.nan
                    else:
                        im_3 = slice_3
                except IndexError:
                    im_3 = np.nan
                if isinstance(im_0, float):
                    pw_0 = np.nan
                else:
                    pw_0 = self.determine_spot_width(im_0, n=n, frac=frac)
                if isinstance(im_1, float):
                    pw_1 = np.nan
                else:
                    pw_1 = self.determine_spot_width(im_1, n=n, frac=frac)
                if isinstance(im_2, float):
                    pw_2 = np.nan
                else:
                    pw_2 = self.determine_spot_width(im_2, n=n, frac=frac)
                if isinstance(im_3, float):
                    pw_3 = np.nan
                else:
                    pw_3 = self.determine_spot_width(im_3, n=n, frac=frac)
                std_min = np.min([pw_0, pw_1, pw_2, pw_3])
                spot_sizes[ind] = std_min

            return spot_sizes
        
    def determine_spots_width_in_window_v2(
        self, image: np.ndarray, spots: np.ndarray, w=5, n=11, mask_fish: Optional[np.ndarray] = None, frac= 0.75):
        """
        Determine spot size based on the half width of the intensity
            in the window around the spot.
        Spot size: nparray of N.2 elements of the form (y, x)
        w: window size
        n: interpolation factor: number of points between pixels.
        """
        spots_c = spots.copy()
        
        if mask_fish is not None:
            sel_spots = []
            for u in range(len(spots)):
                if mask_fish[spots[u][0], spots[u][1]] == 1:
                    sel_spots.append(spots[u])
            spots_c = np.array(sel_spots)
        
        spot_sizes = np.zeros(len(spots_c))
        for ind, tup in enumerate(spots_c):
            y, x = tup
            try:
                slice_0 = image[y - w : y + w + 1, x]
                if slice_0.size == 0:
                    im_0 = np.nan
                else:
                    im_0 = slice_0
            except IndexError:
                im_0 = np.nan
            try:
                slice_1 = image[y, x - w : x + w + 1]
                if slice_1.size == 0:
                    im_1 = np.nan
                else:
                    im_1 = slice_1
            except IndexError:
                im_1 = np.nan
            try:
                slice_2 = []
                for u in range(w + 1):
                    slice_2.append(image[y - w + u, x - w + u])
                for u in range(1, w + 1):
                    slice_2.append(image[y + u, x + u])
                slice_2 = np.array(slice_2)
                if slice_2.size == 0:
                    im_2 = np.nan
                else:
                    im_2 = slice_2
            except IndexError:
                im_2 = np.nan
            try:
                slice_3 = []
                for u in range(w + 1):
                    slice_3.append(image[y + w - u, x - w + u])
                for u in range(1, w + 1):
                    slice_3.append(image[y - u, x + u])
                slice_3 = np.array(slice_3)
                if slice_3.size == 0:
                    im_3 = np.nan
                else:
                    im_3 = slice_3
            except IndexError:
                im_3 = np.nan
            if isinstance(im_0, float):
                pw_0 = np.nan
            else:
                pw_0 = self.determine_spot_width(im_0, n=n, frac= frac)
            if isinstance(im_1, float):
                pw_1 = np.nan
            else:
                pw_1 = self.determine_spot_width(im_1, n=n, frac=frac)
            if isinstance(im_2, float):
                pw_2 = np.nan
            else:
                pw_2 = self.determine_spot_width(im_2, n=n, frac=frac)
            if isinstance(im_3, float):
                pw_3 = np.nan
            else:
                pw_3 = self.determine_spot_width(im_3, n=n,frac=frac)
            std_min = np.min([pw_0, pw_1, pw_2, pw_3])
            spot_sizes[ind] = std_min

        return spot_sizes
    
    
    def extract_diagonals(self, image, x, y):
        # Get the dimensions of the image
        rows, cols = image.shape
        
        # Extract the main diagonal (top-left to bottom-right)
        main_diag = []
        i, j = x, y
        while i >= 0 and j >= 0:
            main_diag.insert(0, image[i, j])
            i -= 1
            j -= 1
        i, j = x + 1, y + 1
        while i < rows and j < cols:
            main_diag.append(image[i, j])
            i += 1
            j += 1
        
        # Extract the anti-diagonal (top-right to bottom-left)
        anti_diag = []
        i, j = x, y
        while i >= 0 and j < cols:
            anti_diag.insert(0, image[i, j])
            i -= 1
            j += 1
        i, j = x + 1, y - 1
        while i < rows and j >= 0:
            anti_diag.append(image[i, j])
            i += 1
            j -= 1
        
        return main_diag, anti_diag
    
    def determine_spots_width_frac_signal(
            self, image: np.ndarray, spots: np.ndarray, h_window=11, size_median_filt=4,frac= 0.5
        ):
            """
            Determine spot size based on a fraction of the intensity maximum
                in the window around the spot.
            image: 3D or 2D np array.
            spots: N.3 or N.2 np array.    
            Spot size: nparray of N.2 elements of the form (y, x): twice the value of the radius.
            h_window: window half size.
            size_median_filt:  windows size used to smoothen the signal. Defaults to 4.
            n: interpolation factor: number of points between pixels.
            frac: fraction of the peak.
            
            returns:
            widths: of the same dimension as spots. Set to nan when the spot is too close to the border.
            """
            spot_sizes   = np.zeros(len(spots))
            widths       = np.full((len(spots),1), np.nan) 
            
            if len(spots[0]) == 2:
                dim_y, dim_x = np.shape(image) 
                for ind, tup in enumerate(spots):
                    y, x = tup
                    if (y - h_window < 0) or (x - h_window < 0):
                        continue
                    elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                        continue
                    else:   
                        signal_h =  image[y-h_window:y+h_window+1,x]
                        signal_v =  image[y,x-h_window:x+h_window+1]
                        sub_image = image[y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                        signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                        
                        w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                        w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                        w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac=frac)
                        w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac=frac)
                        # if necessary (for matplotlib) divide the width by two to obtain the radius
                        # in napari the size is the diameter = width.
                        width = np.median([w0, w1, w2, w3])   
                        widths[ind] = width
                return widths        
            elif len(spots[0]) == 3:
                dim_z, dim_y, dim_x = np.shape(image) 
                for ind, tup in enumerate(spots):
                    z, y, x = tup
                    if (y - h_window < 0) or (x - h_window < 0):
                        continue
                    elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                        continue
                    else:   
                        signal_h =  image[z, y-h_window:y+h_window+1,x]
                        signal_v =  image[z, y,x-h_window:x+h_window+1]
                        sub_image = image[z, y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                        signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                        
                        w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                        w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                        w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac= frac)
                        w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac= frac)
                        width = np.median([w0, w1, w2, w3])   
                        # if necessary (for matplotlib) divide the width by two to obtain the radius
                        # in napari the size is the diameter = width.
                        widths[ind] = width
                return widths              
            else:
                raise DimensionTooHighError(f"Dimension of array is too high (max allowed: 3)")               
        
    
    def determine_spots_width_frac_signal_df(self, im_rna: np.ndarray, df: np.ndarray, h_window: int, size_median_filt: int, frac: float):
        
        df_spots_w_frac_sign               = df.copy()
        if len(df_spots_w_frac_sign):
            df_spots_w_frac_sign['spot_width'] = np.nan
            
            if 'Z' in list(df.columns) and im_rna.ndim == 3:
                df_spots_w_frac_sign.loc[df['in_mask'], 'spot_width'] = df_spots_w_frac_sign[df_spots_w_frac_sign['in_mask']].apply(lambda row: self.determine_spot_width_frac_sign(im_rna,
                                                                                                                                                                                h_window, 
                                                                                                                                                                                size_median_filt,
                                                                                                                                                                                frac,
                                                                                                                                                                                int(row['X']),
                                                                                                                                                                                int(row['Y']),
                                                                                                                                                                                z= int(row['Z'])),
                                                                                                                                                                                axis=1)        
            elif 'Z' not in list(df.columns) and im_rna.ndim == 2:   
                df_spots_w_frac_sign.loc[df['in_mask'], 'spot_width'] = df_spots_w_frac_sign[df_spots_w_frac_sign['in_mask']].apply(lambda row: self.determine_spot_width_frac_sign(im_rna,
                                                                                                                                                                                    h_window, 
                                                                                                                                                                                    size_median_filt,
                                                                                                                                                                                    frac,
                                                                                                                                                                                    int(row['X']),
                                                                                                                                                                                    int(row['Y'])),
                                                                                                                                                                                    axis=1)
        else:
            df_spots_w_frac_sign['spot_width'] = []
        return df_spots_w_frac_sign
        
        
    def determine_spot_width_frac_sign(self, image: np.ndarray,  h_window: int, size_median_filt: int, frac: float, x: int, y: int, z= None):
         
        if z is None: 
            dim_y, dim_x = np.shape(image) 
            if (y - h_window < 0) or (x - h_window < 0):
                return np.nan
            elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                return np.nan
            else:   
                signal_h =  image[y-h_window:y+h_window+1,x]
                signal_v =  image[y,x-h_window:x+h_window+1]
                sub_image = image[y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                
                w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac=frac)
                w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac=frac)
                # if necessary (for matplotlib) divide the width by two to obtain the radius
                # in napari the size is the diameter = width.
                width = np.median([w0, w1, w2, w3])   
                return width        
        else:
            dim_z, dim_y, dim_x = np.shape(image) 
            if (y - h_window < 0) or (x - h_window < 0):
                return np.nan
            elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                return np.nan
            else:   
                signal_h =  image[z, y-h_window:y+h_window+1,x]
                signal_v =  image[z, y,x-h_window:x+h_window+1]
                sub_image = image[z, y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                
                w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac=frac)
                w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac=frac)
                # if necessary (for matplotlib) divide the width by two to obtain the radius
                # in napari the size is the diameter = width.
                width = np.median([w0, w1, w2, w3])   
                return width     
             
    def det_spot_width_frac_peak(self, signal: np.ndarray, size_median_filt=4, frac=0.5):
        """_summary_

        Args:
            signal (np.ndarray): 1D slice vertical, horizontal or diagonal of a spot.
            size_median_filt (int, optional): windows size used to smoothen the signal. Defaults to 4.

        Returns:
            int: width of the bell shaped signal.
        """
        fac_int         = 10
        signal_int      = self.interpolate_signal(signal, n=fac_int)
 
        
        signal_sym      = (signal_int + np.flip(signal_int))/2
        filtered_signal = median_filter(signal_sym, size=size_median_filt*fac_int)
        
        diff_max        = np.max(filtered_signal)- np.min(filtered_signal)
                        
        signal_higher = filtered_signal > frac*diff_max + np.min(filtered_signal)
        ind_max       = np.argmax(signal_higher)
        
        ind = ind_max
        list_ind_higher = []
        while ind < len(signal_higher) and signal_higher[ind]:
            list_ind_higher.append(ind)
            ind = ind+1
        
        ind = ind_max-1
        while ind >=0 and signal_higher[ind]:
            list_ind_higher.append(ind)
            ind = ind-1
            
        width = len(list_ind_higher)/fac_int
        
        return width

    def determine_spots_width_frac_area(
            self, image: np.ndarray, spots: np.ndarray, h_window=11, size_median_filt=4, frac= 0.5
        ):
            """
            Determine spot size based on the half width of the intensity
                in the window around the spot.
            Spot size: nparray of N.2 elements of the form (y, x): twice the value of the radius.
            h_window: window half size
            n: interpolation factor: number of points between pixels.
            frac: fraction of the peak.
            """
            spot_sizes   = np.zeros(len(spots))
            widths       = np.zeros((len(spots),1)) 
            dim_y, dim_x = np.shape(image) 

            if len(spots[0]) == 2:
                for ind, tup in enumerate(spots):
                    y, x = tup
                    if (y - h_window < 0) or (x - h_window < 0):
                        continue
                    elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                        continue
                    else:   
                        signal_h =  image[y-h_window:y+h_window+1,x]
                        signal_v =  image[y,x-h_window:x+h_window+1]
                        sub_image = image[y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                        signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                        
                        w0 = self.det_spot_width_frac_area(signal_h, size_median_filt= size_median_filt, frac= frac)
                        w1 = self.det_spot_width_frac_area(signal_v, size_median_filt= size_median_filt, frac= frac)
                        w2 = self.det_spot_width_frac_area(signal_d1, size_median_filt= size_median_filt, frac=frac)
                        w3 = self.det_spot_width_frac_area(signal_d2, size_median_filt= size_median_filt, frac=frac)
                        
                        width = np.median([w0, w1, w2, w3])   
                        # if necessary (for matplotlib) divide the width by two to obtain the radius
                        # in napari the size is the diameter = width.
                        widths[ind] = width
                return widths        
            elif len(spots[0]) == 3:        
                for ind, tup in enumerate(spots):
                    z, y, x = tup
                    if (y - h_window < 0) or (x - h_window < 0):
                        continue
                    elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                        continue
                    else:   
                        signal_h =  image[z, y-h_window:y+h_window+1,x]
                        signal_v =  image[z, y,x-h_window:x+h_window+1]
                        sub_image = image[z, y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                        signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                        
                        w0 = self.det_spot_width_frac_area(signal_h, size_median_filt= size_median_filt, frac= frac)
                        w1 = self.det_spot_width_frac_area(signal_v, size_median_filt= size_median_filt, frac= frac)
                        w2 = self.det_spot_width_frac_area(signal_d1, size_median_filt= size_median_filt, frac= frac)
                        w3 = self.det_spot_width_frac_area(signal_d2, size_median_filt= size_median_filt, frac= frac)
                        width = np.median([w0, w1, w2, w3])   
                        # if necessary (for matplotlib) divide the width by two to obtain the radius
                        # in napari the size is the diameter = width.
                        widths[ind] = width
                return widths              
            else:
                raise DimensionTooHighError(f"Dimension of array is too high (max allowed: 3)")               
        
    def det_spot_width_frac_area(self, signal: np.ndarray, size_median_filt=4, frac=0.5):
        """_summary_

        Args:
            signal (np.ndarray): 1D slice vertical, horizontal or diagonal of a spot.
            size_median_filt (int, optional): windows size used to smoothen the signal. Defaults to 4.

        Returns:
            int: width of the bell shaped signal.
        """
        fac_int         = 10
        signal_int      = self.interpolate_signal(signal, n=fac_int)
 
        
        signal_sym      = (signal_int + np.flip(signal_int))/2
        filtered_signal = median_filter(signal_sym, size=size_median_filt*fac_int)
        
        filtered_signal_sub = filtered_signal - np.min(filtered_signal)
        area            = np.sum(filtered_signal_sub)
        threshs         = np.linspace(np.min(filtered_signal_sub), np.max(filtered_signal_sub), 100)
        
        ind       = -1
        area_meas = 0
        while area_meas < area*frac and len(threshs) + ind >=0:
            thresh    = threshs[ind]
            area_meas = np.sum(filtered_signal_sub[filtered_signal_sub > thresh])
            ind       = ind - 1
        signal_higher = filtered_signal_sub > threshs[ind]
        ind_max       = np.argmax(filtered_signal_sub)
        
        ind2 = ind_max
        list_ind_higher = []
        while ind2 < len(signal_higher) and signal_higher[ind2]:
            list_ind_higher.append(ind2)
            ind2 = ind2+1
        
        ind2 = ind_max-1
        while ind2 >=0 and signal_higher[ind2]:
            list_ind_higher.append(ind2)
            ind2 = ind2-1
            
        width = len(list_ind_higher)/fac_int 
         
        return width
   
    def find_all_contours(self, masks: np.ndarray):
        """
        Find contours on a labeled mask. Each mask is taken separately
        and its countours are determined.

        Input:
            masks:  np.ndarray. Each mask has a unique label (integer).

        Output:
            list of contours (np.ndarrays).
        """
        contours_method = []
        padded_masks    = np.pad(masks, pad_width=1, mode='constant', constant_values=0)        

        for mask_temp_num in np.unique(masks):
            if mask_temp_num:
                mask_temp = (padded_masks == mask_temp_num) * 1
                contours_method.append(
                    measure.find_contours(mask_temp.astype(np.uint8), 0.5)
                )
        contours_method = [item for sublist in contours_method for item in sublist]
        return contours_method

    def remove_labels_from_masks(self, masks: np.ndarray, label_list: Union[np.ndarray, int]) -> np.ndarray:
        """Remove the labels from the masks.

        Args:
            masks (np.nparray): each mask has a unique label (int).
            label_list (np.ndarray): list of labels to remove.

        Returns:
            np.ndarray: masks with the labels removed
        """
        masks_rem = masks.copy()
        if isinstance(label_list, np.ndarray):
            if len(label_list):
                for lab in label_list:
                    masks_rem[masks_rem==lab] = 0
        return masks_rem  

    def match_nuc_cell(self, nuc_label: np.ndarray, cell_label: np.ndarray):
        """Match each nucleus instance with the most overlapping cell instance.
        
        In this version:
        - nuclei without cells are discarded.
        - cells without nuclei are discarded.
        - when there are more than one nuclei in a cell, they are labeled with the same label.
        - the cell mask is expanded to include all overlapping nuclei.
        - conflicts between nuclei overlaping onto another cell body are solved. 
        Parameters
        ----------
        nuc_label : np.ndarray
            Labelled image of nuclei with shape (z, y, x) or (y, x).
        cell_label : np.ndarray
            Labelled image of cells with shape (z, y, x) or (y, x).

        Returns
        -------
        new_nuc_label : np.ndarray
            Labelled image of nuclei with shape (z, y, x) or (y, x).
        new_cell_label : np.ndarray
            Labelled image of cells with shape (z, y, x) or (y, x).
        """
         
        nuc_label_c  = nuc_label.copy()
        cell_label_c = cell_label.copy()

        new_nuc_label  = np.zeros_like(nuc_label_c)
        new_cell_label = cell_label_c.copy()
        
        # loop over nuclei
        for i_nuc in range(1, nuc_label_c.max() + 1):
            nuc_mask = nuc_label_c == i_nuc
        
            if nuc_mask.sum() == 0:
                continue
        
            i_cell = np.argmax(np.bincount(cell_label_c[nuc_mask]))  # check if a cell is labelled with this value
            if i_cell != 0:
                new_nuc_label[nuc_mask] = i_cell # assign it the label of the cell
        
        # loop over cells, remove the ones with no nucleus , and do logical or with masks (the cell mask is expanded to include all overlapping nuclei)
        for i_cell in range(1, cell_label_c.max() + 1):
            cell_mask = cell_label_c == i_cell
        
            if cell_mask.sum() == 0:
                continue
        
            if (new_nuc_label == i_cell).sum() == 0:
                new_cell_label[new_cell_label == i_cell] = 0
                continue   

            nuc_mask = new_nuc_label == i_cell
            new_cell_mask = np.logical_or(cell_mask, nuc_mask)
            new_cell_label[new_cell_mask] = i_cell
        
        # conflicts between nuclei overlaping onto another cell body are solved.
        for j_nuc in range(1, new_nuc_label.max() + 1):
            nuc_mask_j = new_nuc_label == j_nuc
        
            if nuc_mask_j.sum() == 0:
                continue
            
            for i_cell in range(1, new_cell_label.max() + 1):
                if i_cell != j_nuc:
                    cell_mask_i = new_cell_label == i_cell
                    if cell_mask_i.sum() == 0:
                        continue    
                
                    intersect = cell_mask_i*nuc_mask_j
                    if intersect.sum() != 0:
                        
                        mask_cell_j = new_cell_label == j_nuc
                        mask_cell_j = np.logical_or(mask_cell_j, nuc_mask_j) 
                        cell_mask_i = np.logical_and(cell_mask_i, np.logical_not(nuc_mask_j))

                        new_cell_label[mask_cell_j] = j_nuc
                        new_cell_label[cell_mask_i] = i_cell
            
        return  new_nuc_label, new_cell_label 
    
    
class DimensionTooHighError:
    pass    