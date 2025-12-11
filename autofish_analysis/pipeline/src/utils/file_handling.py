import os
import numpy as np
import pandas as pd
from nd2reader import ND2Reader
from skimage.io import imsave
from skimage import io
import tkinter as tk
from tkinter import filedialog
import warnings
from pathlib import Path
import readlif as readlif
from readlif.reader import LifFile
from typing import Union, List

class FileProcessor:
    @staticmethod
    def read_convert_np(file_path):
        "Read nd2 file, convert to two numpy arrays :(FOV,x,Z,H,W), x = 2 dims, respectively DAPI and GFP"
        with ND2Reader(file_path) as images:
            metadata = images.metadata
            dapi_index = metadata["channels"].index("DAPI")
            gfp_index = metadata["channels"].index("GFP")

            dapi_images = np.empty((1, metadata["height"], metadata["width"]))
            gfp_images = np.empty((1, metadata["height"], metadata["width"]))

            for field_of_view in range(len(metadata["fields_of_view"])):
                for z_level in range(len(metadata["z_levels"])):
                    dapi_image = np.array(images[field_of_view, dapi_index, z_level])[
                        dapi_index, :, :
                    ]
                    dapi_image = np.expand_dims(dapi_image, axis=0)
                    dapi_images = np.concatenate((dapi_images, dapi_image), axis=0)

                    gfp_image = np.array(images[field_of_view, dapi_index, z_level])[
                        gfp_index, :, :
                    ]
                    gfp_image = np.expand_dims(gfp_image, axis=0)
                    gfp_images = np.concatenate((gfp_images, gfp_image), axis=0)

            dapi_images = dapi_images[1:, :, :]
            gfp_images = gfp_images[1:, :, :]

            dapi_images = np.expand_dims(np.expand_dims(dapi_images, axis=0), axis=0)
            gfp_images = np.expand_dims(np.expand_dims(gfp_images, axis=0), axis=0)

            return dapi_images, gfp_image

    def read_lif_many_images(self, file_path):
        """
            Read a LIF file with several images.
            
            Input: file_path: string
            
            Output: 
                list of image names
                list of np.array images (c,z,y,x)
        """
        file = LifFile(file_path)
        n_images = len(file.image_list)
        image_list = []
        images_names = []
        for im_n in range(n_images):
            lif_image = file.get_image(im_n)  
            stacked_images_c = self.read_lif_with_channels(lif_image)
            name             = lif_image.name
            images_names.append(name)
            image_list.append(stacked_images_c)
            
        return images_names, image_list
            
    def read_lif_with_channels(self, lif_image: readlif.reader.LifImage):
        """
        Stacks images from a lif_image object along the z-axis.

        Args:
        lif_image: The lif_image object containing the images.

        Returns:
        A NumPy array with dimensions (c, z, y, x) containing the stacked images.
        """
        num_z_slices = lif_image.info['dims'].z
        num_channels = lif_image.info['channels']

        # Create an empty array to store the stacked images for all channels
        stacked_images = np.empty((num_channels, num_z_slices, lif_image.get_frame(z=0, c=0).size[0], lif_image.get_frame(z=0, c=0).size[1]), dtype=np.array(lif_image.get_frame(z=0, c=0)).dtype)

        # Iterate through channels and z-slices to append images to the stack
        for c in range(num_channels):
            for z in range(num_z_slices):
                pil_image = lif_image.get_frame(z=z, c=c)
                np_image = np.array(pil_image)
                stacked_images[c, z, :, :] = np_image

        return stacked_images
        


    @staticmethod
    def delete_temp_files(temp_dir: str):
        "Deletes all files in the 'temp_dir' directory, except for '.gitkeep'."

        if not os.path.exists(temp_dir):
            print(f"Directory '{temp_dir}' does not exist.")
            return

        for file in os.listdir(temp_dir):
            if file != ".gitkeep":
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
       
    def select_and_load_file(self):
        root = tk.Tk()
        root.withdraw()
        root.geometry("500x300")
        file_path = filedialog.askopenfilename(
            title="Select a file in the Acquisition folder"
        )
        if file_path:
            FILE_NAME = file_path.split("/")[-1][:-4]
            FOLDER_NAME = str(Path(file_path).parent)
            FILE_NAME_no_spaces = FILE_NAME.replace(" ", "")
            if FILE_NAME != FILE_NAME_no_spaces:
                print(
                    "remove all spaces from file name ! saving temp file names without space."
                )
                FILE_NAME = FILE_NAME_no_spaces
            im = io.imread(file_path)
            print("File loaded")
            return im, FILE_NAME, FOLDER_NAME
        else:
            print("File not loaded properly, start again")
            return None, None, None
        
    
    def select_folder(self, initialdir,  title=None,):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.geometry("500x300")
        if title is not None:
            title = title
        else:
            title = "Select a folder"
            
        folder_path = filedialog.askdirectory(
             initialdir=initialdir, title=title,
        )
        if folder_path:
            print(f"Selected folder: {folder_path}")
            root.destroy() # Clean up tkinter instance
            return folder_path
        else:
            print("No folder selected.")
            root.destroy() # Clean up tkinter instance
            return None

    def select_file(self, initialdir=None, title=None):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.geometry("500x300")
        if title is not None:
            title = title
        else:
            title = "Select a file"   
        file_path = filedialog.askopenfilename(
             initialdir=initialdir, title=title)
        
        if file_path:
            print(f"Selected file: {file_path}")
            root.destroy() # Clean up tkinter instance
            return file_path
        else:
            print("No file selected.")
            root.destroy() # Clean up tkinter instance
            return None

    def read_tsv_files_dw_dots(self, file_tsv: str):
        if os.path.exists(file_tsv):
            df = pd.read_csv(file_tsv, sep="\t")
            df = df[df["use"] == 1]
            df = df[["z", "y", "x"]]
            return df.to_numpy().astype(int)
        else:
            raise ImportError(f"Inexistent file {file_tsv}")
        
        
    def find_tif_files(self, root_folder):
        """
        Finds all .tif files within a given root folder and its subfolders.

        Args:
            root_folder: The path to the root folder to search.

        Returns:
            A list of paths to all .tif files found.
        """
        tif_files = []
        for path in Path(root_folder).rglob('*.tif'):
            tif_files.append(path)
        return tif_files    

    def save_spots_distributed_files(self, numpy_file_add: str, format: str, dict_data: dict, im_dim: int):
        folder_add = Path(numpy_file_add).parent
        if format == 'spots_IDzyx':
            dict_address = {}
            for key, arr in dict_data.items():
                csv_filepath      = folder_add / Path(key + '.csv')
                dict_address[key] = csv_filepath
                if len(arr):
                    if isinstance(arr, np.ndarray):         # first time : creation of the labels
                        arr_ext = np.hstack((np.arange(1, len(arr)+1).reshape(-1, 1), arr))
                        if im_dim == 3:
                            df = pd.DataFrame(data=arr_ext, columns=['RNA_Spot_ID', 'Z','Y','X'])
                            df.to_csv(csv_filepath, index=False)
                        elif im_dim == 2:
                            df = pd.DataFrame(data=arr_ext, columns=['RNA_Spot_ID', 'Y','X'])
                            df.to_csv(csv_filepath, index=False)
                        else:
                            print('Expected images should have 2 or 3 dimensions !')
                    elif isinstance(arr, pd.DataFrame):
                        arr.to_csv(csv_filepath, index=False)
                else:
                    if im_dim == 3:
                        df = pd.DataFrame(data=[], columns=['Z','Y','X'])
                        df.to_csv(csv_filepath, index=False)
                    elif im_dim == 2:
                        df = pd.DataFrame(data=[], columns=['Y','X'])
                        df.to_csv(csv_filepath, index=False)
            np.save(numpy_file_add, dict_address)               
        else:
            print('unknown format, existing format: spots_IDzyx')   
           
    def load_spots_distributed_files(self, dict_address_path: str, format: str):
        "returns a dict of dataframes"        
        dict_address = np.load(dict_address_path, allow_pickle=True)[()]
        if format == 'spots_IDzyx':
            dict_data = {}
            for key, address in dict_address.items():                    
                    df = pd.read_csv(address)
                    dict_data[key] = df  
            return dict_data
        else:
            print('unknown format, existing format: spots_IDzyx') 
             
    def save_masks_distributed_files(self, dict_address_path: str, dict_data: dict):
        folder_add = Path(dict_address_path).parent
        dict_address = {}
        
        for key, arr in dict_data.items():
            png_filepath      = folder_add / Path(key + '.png')
            dict_address[key] = png_filepath
            imsave(png_filepath, arr.astype(np.uint16))
        np.save(dict_address_path, dict_address)
        
    def load_masks_distributed_files(self, dict_address_path: str):
        dict_address = np.load(dict_address_path, allow_pickle=True)[()]
        dict_data = {}
        for key, address in dict_address.items():
            im  = io.imread(address)
            dict_data[key] = im
        return dict_data    
        
    def save_masks_stats_distributed_files_init(self, numpy_file_add: str,  mask_path: str, col_name='Cell_ID'):
        """Load the masks and save in a file given by the user a numpy file and the .csv files. 

        Args:
            numpy_file_add (str): file address of the numpy file (containing the key/addresses).
            mask_cell_path (str): file address with the numpy file referencing the addresses of the masks.
        """
        
        dict_mask = self.load_masks_distributed_files(mask_path)     # load the masks 

        dict_labels = {}
        for name in list(dict_mask.keys()):
            nums_list = np.unique(dict_mask[name])                   # repertoriate the masks numbers
            if 0 in nums_list:
                nums_list = nums_list[nums_list != 0]
            
            dict_labels[name] = nums_list
        
        folder_add = Path(numpy_file_add).parent
        dict_address = {}
        
        for key, masks_labs in dict_labels.items():
            csv_filepath      = folder_add / Path(key + '.csv')
            dict_address[key] = csv_filepath
            if len(masks_labs):
                if isinstance(masks_labs, np.ndarray):               # first time : creation of the labels
                    masks_labs = masks_labs.reshape(-1, 1)
                    
                    df = pd.DataFrame(data=masks_labs, columns=[col_name])
                    df.to_csv(csv_filepath, index=False)

                elif isinstance(masks_labs, pd.DataFrame):
                    masks_labs.to_csv(csv_filepath, index=False)    
            else:
                df = pd.DataFrame(data=[], columns=[col_name])
                df.to_csv(csv_filepath, index=False)

        np.save(numpy_file_add, dict_address)     
        
        
        
    def save_masks_stats_distributed_files_modif(self, file_add_mask_stats,  dic_masks_cell_stats):
        
        dic_mask_stats = np.load(file_add_mask_stats, allow_pickle=True)[()]
        for key, add in  dic_mask_stats.items():
            dic_masks_cell_stats[key].to_csv(add, index=False)
        
    def load_pd_distributed_files(self, dict_address_path: str):
        "returns a dict of dataframes"        
        dict_address = np.load(dict_address_path, allow_pickle=True)[()]
        dict_data = {}
        for key, address in dict_address.items():                    
                df = pd.read_csv(address)
                dict_data[key] = df  
        return dict_data  