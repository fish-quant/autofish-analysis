import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorsys

from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.figure import Figure as figure

import matplotlib.colors as mcolors
from napari.utils.colormaps import Colormap

class Plots:
    @staticmethod
    def plot_cell_nuclei_cytoplasm(
        list_masks: list, figsize=(10, 20), generate_fig=True
    ):
        tot_im = np.zeros_like(list_masks[0][0])
        for ind_cell, cell in enumerate(list_masks):
            tot_im = (
                tot_im
                + cell[2] * (0.3 + 0.7 * np.random.rand())
                + cell[1] * (0.3 + 0.7 * np.random.rand())
            )
        if generate_fig:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.imshow(tot_im)
            ax.set_title("cell and cytoplasm segmentation")
            ax.set_axis_off()
            return None
        else:
            return tot_im

    def generate_colors_hsv(self, n):
        "Generate np array of n times 3 colors"
        colors = []
        for _ in range(n):
            h, s, v = np.random.rand(3)
            rgb = colorsys.hsv_to_rgb(h, s, v)
            colors.append(rgb)
        return np.array(colors)

    @staticmethod
    def create_mask_detection(
        rna_mip: np.ndarray, spots, figsize=(10, 20), generate_fig=True
    ):
        """
        Creates a mask, in which detected fish positions are set to 1.
        """
        rna_mip_im = np.zeros(np.shape(rna_mip))  # .copy()
        for i, coordinates in enumerate(spots):
            y, x = coordinates[1], coordinates[2]
            rna_mip_im[y, x] = 1

        if generate_fig:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(rna_mip_im)
            ax.set_axis_off()
            return None
        else:
            return rna_mip_im

    def plot_fitted_sizes(self, image: np.ndarray, spots: np.ndarray, spot_sizes: np.ndarray, figsize=(15, 13), color= 'red', show= False):
        """
            Spot size: nparray of N.2 elements of the form (y, x)
        """
        fig, ax = plt.subplots(figsize = figsize)
        ax.imshow(image, cmap='Greys')
        for u in range(spots.shape[0]):
            if not np.isnan(spot_sizes[u]):
                circle = Circle((spots[u, 1], spots[u, 0]), spot_sizes[u], color=color, fill=False) 
                ax.add_patch(circle)
        if show:
            plt.show()
        else:
            return fig, ax
           
    def plot_fitted_sizes_sub_ax(self, image: np.ndarray, spots: np.ndarray, spot_sizes: np.ndarray, 
                                 fig: figure, axes: np.ndarray, lign: int, col: int,
                                 leg_x: list, leg_y: list, title: str, color='red', fill=True
                                 ):
        """
            Spot size: nparray of N.2 elements of the form (y, x)
        """
        axes[lign, col].imshow(image, cmap='Greys')

        for u in range(spots.shape[0]):
            if not np.isnan(spot_sizes[u]):
                circle = Circle((spots[u, 1], spots[u, 0]), spot_sizes[u], color=color, fill=False) 
                axes[lign, col].add_patch(circle)
     
        axes[lign, col].spines['top'].set_visible(False)
        axes[lign, col].spines['right'].set_visible(False)
        axes[lign, col].spines['bottom'].set_visible(False)
        axes[lign, col].spines['left'].set_visible(False)
        axes[lign, col].tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
        if col==0:
            axes[lign, col].set_ylabel(leg_y[lign])
        if lign==2:
            axes[lign, col].set_xlabel(leg_x[col])
        if col==0 and lign==0:   
            axes[lign, col].set_title(title)   
                   
      
    def plot_dots_fix_size(self, image: np.ndarray, spots: np.ndarray, fig: figure, axes: np.ndarray, lign: int, col: int,
                          spot_size: float, leg_x: list, leg_y: list, title: str, color='red', fill=True):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        axes[lign, col].imshow(image, cmap='Greys')
        
        axes[lign, col].spines['top'].set_visible(False)
        axes[lign, col].spines['right'].set_visible(False)
        axes[lign, col].spines['bottom'].set_visible(False)
        axes[lign, col].spines['left'].set_visible(False)
        axes[lign, col].tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
        if col==0:
            axes[lign, col].set_ylabel(leg_y[lign])
        if lign==2:
            axes[lign, col].set_xlabel(leg_x[col])
        if col==0 and lign==0:   
            axes[lign, col].set_title(title)
                    
        for u in range(spots.shape[0]):
            if not np.isnan(spots[u, 0]) and not np.isnan(spots[u, 1]):
                circle = Circle((spots[u, 1], spots[u, 0]), spot_size, color=color, fill=fill)
                axes[lign, col].add_patch(circle)

    def plot_dots_fix_size_v2(self, image: np.ndarray, spots: np.ndarray, fig: figure, axes: np.ndarray, lign: int, col: int,
                          spot_size: float, leg_x: list, leg_y: list, title: str, color='red', fill=True, mask_fish=None):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        axes[lign, col].imshow(image, cmap='Greys')
        
        axes[lign, col].spines['top'].set_visible(False)
        axes[lign, col].spines['right'].set_visible(False)
        axes[lign, col].spines['bottom'].set_visible(False)
        axes[lign, col].spines['left'].set_visible(False)
        axes[lign, col].tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
        if col==0:
            axes[lign, col].set_ylabel(leg_y[lign])
        if lign==2:
            axes[lign, col].set_xlabel(leg_x[col])
        if col==0 and lign==0:   
            axes[lign, col].set_title(title)
        
        if mask_fish is None:            
            for u in range(spots.shape[0]):
                if not np.isnan(spots[u, 0]) and not np.isnan(spots[u, 1]):
                    circle = Circle((spots[u, 1], spots[u, 0]), spot_size, color=color, fill=fill)
                    axes[lign, col].add_patch(circle)
        else:
            for u in range(spots.shape[0]):
                if not np.isnan(spots[u, 0]) and not np.isnan(spots[u, 1]):
                    if mask_fish[int(spots[u, 0]), int(spots[u, 1])]:
                        circle = Circle((spots[u, 1], spots[u, 0]), spot_size, color=color, fill=fill)
                        axes[lign, col].add_patch(circle)
                    else:
                        circle = Circle((spots[u, 1], spots[u, 0]), spot_size, color='c', fill=fill)
                        axes[lign, col].add_patch(circle)     
            
            
            
            
    def plot_hist_sizes(self, spot_sizes: np.ndarray, 
                        fig: figure, axes: np.ndarray, lign: int, col: int,
                        leg_x: list, leg_y: list, title: str, bins = 50, color='red', color_mean= '#FF9D23',
                        fill=True, shape=None):
        
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
                
        spot_sizes_not_nan = spot_sizes[np.logical_not(np.isnan(spot_sizes))]
        
        mean = np.mean(spot_sizes_not_nan)
        axes[lign, col].hist(spot_sizes_not_nan, bins = bins, color=color)
        n, bins_t = np.histogram(spot_sizes_not_nan, bins=bins)        
        
        if lign == 0 and col ==max_col:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2, label= 'mean')
            axes[lign, col].legend(loc='upper right', fontsize=6)
        else:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2)
        
        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[lign])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)    
        if lign==max_lign:
            if isinstance(leg_x, list):
                axes[lign, col].set_xlabel(leg_x[col])
            elif isinstance(leg_x, str):
                axes[lign, col].set_xlabel(leg_x)    
        if col==0 and lign==0:
                axes[lign, col].set_title(title)
                    
        mean_r = np.floor(mean*10)/10
        xticks = axes[lign, col].get_xticks()
        xticklabels  = axes[lign, col].get_xticklabels()
        xtickslabels = [label.get_text() for label in xticklabels]

        xticks = np.append(xticks, mean_r)
        xtickslabels.append(str(mean_r))

        axes[lign, col].set_xticks(xticks)
        axes[lign, col].set_xticklabels(xtickslabels)
        xticklabels = axes[lign, col].get_xticklabels()

        xticklabels[-1].set_color(color_mean) 
        axes[lign, col].set_xlim([0, bins_t[-1]])
        
        axes[lign, col].tick_params(axis='x', rotation=45)



    def plot_hist_num_dots_nuclei(self, count_spots: np.ndarray, fig: figure, axes: np.ndarray, lign: int, col: int,
                                  leg_x: list, leg_y: list, title: str, color='red', color_mean= '#FF9D23', bins= 50,
                                  struct = 'nuclei', shape = None):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        mean = np.median(count_spots)
        axes[lign, col].hist(count_spots, bins = bins, color=color)
        n, bins_t = np.histogram(count_spots, bins=bins)        
        
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
        
        if lign == 0 and col ==max_col:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2, label= 'median spot/'+struct)
            axes[lign, col].legend(loc='upper right', fontsize=6)
        else:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2)

        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[lign])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[col])
        elif isinstance(leg_x, str):
            axes[lign, col].set_xlabel(leg_x)
                
        if col==0 and lign==0:   
            axes[lign, col].set_title(title + '   # spots / '+ struct )
                    
        mean_r = np.floor(mean * 10) / 10
        xticks = axes[lign, col].get_xticks()
        xticklabels  = axes[lign, col].get_xticklabels()
        xtickslabels = [label.get_text() for label in xticklabels]

        xticks = np.append(xticks, mean_r)
        xtickslabels.append(str(mean_r))

        axes[lign, col].set_xticks(xticks)
        axes[lign, col].set_xticklabels(xtickslabels, rotation=60)
        xticklabels = axes[lign, col].get_xticklabels()

        xticklabels[-1].set_color(color_mean) 
        axes[lign, col].set_xlim([0, bins_t[-1]])
        
        
    def plot_intensities(self, bulk_fish_int: list, fig: figure, axes: np.ndarray, lign: int, col: int,
                                  leg_x: list, leg_y: list, title: str, color='red', color_mean= '#FF9D23', bins= 50, shape=None, max_x=None, num_decimals=0):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
        
        mean = np.mean(bulk_fish_int)
        axes[lign, col].hist(bulk_fish_int, bins = bins, color=color)
        n, bins_t = np.histogram(bulk_fish_int, bins=bins)        
        
        if lign == 0 and col ==max_col:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2, label= 'mean')
            axes[lign, col].legend(loc='upper right', fontsize=6)
        else:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2)
        
        if max_x is not None:
            axes[lign, col].set_xlim([0, max_x])

        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[col])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        #if lign==max_lign:
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[lign])
        elif isinstance(leg_x, str):
            axes[lign, col].set_xlabel(leg_x)

        if col==0 and lign==0:   
            axes[lign, col].set_title(title)
                   
        mean_r = np.round(mean, decimals=num_decimals)

        xticks = axes[lign, col].get_xticks()
        xticklabels  = axes[lign, col].get_xticklabels()
        xtickslabels = [label.get_text() for label in xticklabels]

        xticks = np.append(xticks, mean_r)
        xtickslabels.append(str(mean_r))

        axes[lign, col].set_xticks(xticks)
        axes[lign, col].set_xticklabels(xtickslabels)
        xticklabels = axes[lign, col].get_xticklabels()

        xticklabels[-1].set_color(color_mean) 
        if max_x is None:
            axes[lign, col].set_xlim(0, bins_t[-1])
        
        axes[lign, col].tick_params(axis='x', rotation=45)
        
        
    def plot_subcellular_localization(self, g_spots_in_masks: np.ndarray, fig: figure,
                                      axes: np.ndarray, lign: int, col: int, leg_x: list,
                                      leg_y: list, title: str,  color='red', struct = '   number of rna outside or inside the cells',
                                      shape=None):
        """
        Count the number of rnas inside vs outside cells.
        """    
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
               
        bins = np.array([-0.3, 0.3, 0.7, 1.3]) 
        axes[lign, col].hist(g_spots_in_masks , bins=bins, color=color);
        axes[lign, col].set_xticks([0, 1])
        
        if struct == '   number of rna outside or inside the cells':
            axes[lign, col].set_xticklabels(['Out', 'In']) 
        else:
            axes[lign, col].set_xticklabels(['Cyto', 'Nucleus']) 
        
        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[lign])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        #if lign==max_lign:
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[col])
        else:
            axes[lign, col].set_xlabel(leg_x)
                
        if col==0 and lign==0:   
            axes[lign, col].set_title(title + struct )
        
        #axes[lign, col].tick_params(axis='x', rotation=15)

    def custom_color_map(self, pow=1, nb_points= 256):
        """
        number of points
        pow: between 0 and 1.
        
        fig, ax = plt.subplots(figsize=figsize)
        q = ax.imshow(im_2d_mip_n,  vmin = 0, vmax=1, cmap=custom_cmap)
        cbar = fig.colorbar(q, ax=ax)
        """

        linear_ramp = np.linspace(0, 1, nb_points)
        powered_ramp = linear_ramp ** pow
        colors = np.zeros((nb_points, 4))
        colors[:, 0] = powered_ramp  
        colors[:, 1] = powered_ramp  
        colors[:, 2] = powered_ramp  
        colors[:, 3] = 1.0           # Alpha channel (fully opaque)
        custom_cmap = mcolors.ListedColormap(colors)
        return custom_cmap
    
    def custom_color_map_for_napari(self, pow=1, nb_points= 256):
        """
        number of points
        pow: between 0 and 1.

        Example of usage:
        cm = pts.custom_color_map_for_napari(pow=.5, nb_points= 256)
        viewer = napari.Viewer()
        viewer.add_image(rescaled_image, colormap=('custom_gray', cm))

        """
        linear_ramp = np.linspace(0, 1, nb_points)
        powered_ramp = linear_ramp ** pow
        colors = np.zeros((nb_points, 4))
        colors[:, 0] = powered_ramp  
        colors[:, 1] = powered_ramp  
        colors[:, 2] = powered_ramp  
        colors[:, 3] = 1.0           # Alpha channel (fully opaque)
        return Colormap(colors)
    

    def violin_plot_intensities(self, ints, figsize=(14,3), exp_name=None, rotation=85, names_short= None, color='k', ymin = 0, ymax=None):
                
        fig, ax = plt.subplots(figsize=figsize)
        violinplot = ax.violinplot(ints, showmedians=True, showextrema=False)

        for i, group in enumerate(ints):
            x_coords = np.ones_like(group) * (i + 1) + np.random.normal(0, 0.01, size=len(group))
            for ind in range(len(x_coords)):
                ax.plot(x_coords[ind], group[ind], color=color, marker='.', markersize = 1)
            
        if ymax is not None:
            ax.set_ylim([ymin, ymax])    
            
        ax.set_xlabel("Conditions")
        ax.set_ylabel(f"{exp_name}")
        
        if names_short is not None:
            ax.set_xticks(np.arange(1, len(ints)+1 ))
            ax.set_xticklabels(names_short, rotation=rotation)
        
        return fig            
    
    def bar_plots(self, names: list, dic_signal: dict, dic_std: dict, method: str,color: str): 
    
        fig, ax = plt.subplots(figsize=(14, 5))  # Adjust figure size for better readability
        plt.bar(names, dic_signal, yerr= dic_std, capsize=5, color = color, edgecolor = 'black')
        plt.xlabel("Conditions", fontsize=12)
        plt.ylabel(f"SNR {method}", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize = 10)  # Rotate x-axis labels for better readability if needed
        plt.yticks(fontsize = 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig