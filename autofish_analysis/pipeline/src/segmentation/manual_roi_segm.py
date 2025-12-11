import numpy as np
import napari
from qtpy.QtWidgets import QPushButton
from pathlib import Path
from skimage import io
from skimage.draw import polygon
from skimage.measure import find_contours 
 
class ManualSegmentation():
    
    def convert_shapes_to_mask(self, shapes_layer, image_shape):
        "Convert the actual shapes layer to a binary mask"
        mask = np.zeros(image_shape, dtype=np.uint8)
        for shape in shapes_layer.data:
            vertices = shape[:, :2]
            rr, cc = polygon(vertices[:, 0], vertices[:, 1], mask.shape)
            mask[rr, cc] = 1
        return mask
    
    
    def mask_to_shapes(self, mask):
        "Find contours of a binary mask"
        contours = find_contours(mask, level=0.5)
        shapes = []
        if len(contours):
            for contour in contours:
                vertices = np.array(contour, dtype=np.int32)
                shapes.append(vertices)
        return shapes
    
    
    def fill_mask_and_save(self):
        "If there are shapes in the shapes layer, stores them in the variable disk_mask"        
        if len(self.viewer.layers[2].data):
            mask = self.convert_shapes_to_mask(self.viewer.layers[2], self.im_2d_dimensions)
            self.list_masks[self.current_image_index] = mask
            self.viewer.layers[1].data = mask
            name = Path(self.list_im[self.current_image_index]).stem
            self.dict_masks[name] = mask
        else:
            mask = np.zeros(self.im_2d_dimensions, dtype=np.uint8)
            name = Path(self.list_im[self.current_image_index]).stem
            self.dict_masks[name] = mask
        
    def delete_mask_and_save(self):
        "Delete the shapes layer"
        self.viewer.layers[1].data = np.zeros(self.im_2d_dimensions, dtype=np.uint8)
        self.viewer.layers[2].data = []
        self.fill_mask_and_save()        
            
        
    def __init__(self,
                 batch_images: list,
                 im_2d_dimensions: tuple,
                 ):
        """Manual segmentation of ROIS of interest. Creates a binary map out
        with several possible submasks (all set and filled to 1)

        Args:
            batch_images (list): list of absolute paths to the files
            im_2d_dimensions (tuple): shape of images (original or projected along the Z dimension)
        """
        self.list_im             = batch_images
        self.current_image_index = 0
        self.list_masks          = [np.zeros(im_2d_dimensions, dtype=np.uint8) for u in range(len(batch_images))]
        self.im_2d_dimensions    = im_2d_dimensions
        self.dict_masks          = {}

    def display_next_image(self, button):
        "Stores the current masks and iterates to the next image. If there are no masks, masks=zeros and shape layer is empty"                
        self.fill_mask_and_save() # store first the masks (if they exist)               
        self.current_image_index =  (self.current_image_index + 1) % len(self.list_im) # then load the next image and the masks
        button.setText(f"Image {self.current_image_index}, {Path(self.list_im[self.current_image_index]).stem}")
        if self.list_im[self.current_image_index]:
            self.read_image()
            val = np.percentile(self.curr_im, 99)
            self.viewer.layers[0].data = self.curr_im
            self.viewer.layers[0].contrast_limits = (0, val)
            self.viewer.layers[1].data = self.list_masks[self.current_image_index]
            
            if np.any(self.list_masks[self.current_image_index]):
                self.viewer.layers[2].data = self.mask_to_shapes(self.list_masks[self.current_image_index])
            else:
                self.viewer.layers[2].data = []
            
        else:
            self.viewer.layers[0].data = np.zeros(self.im_2d_dimensions, dtype=np.uint8)
            self.viewer.layers[1].data = np.zeros(self.im_2d_dimensions, dtype=np.uint8)
            self.viewer.layers[2].data = []

                
    def read_image(self):
        "read image, by convention we only use tiff images of 3 or 2 dimensions (Z,Y,X), only one channel."
        im = io.imread(self.list_im[self.current_image_index])
        if im.ndim == 2:
            self.curr_im = im
        elif im.ndim == 3:
            self.curr_im = np.max(im, axis = 0) 
        
    def run(self):
        """
        """
        self.viewer = napari.Viewer()        
        if self.list_im[0]:
            self.read_image()
            val = np.percentile(self.curr_im, 99)
            self.viewer.add_image(self.curr_im, rgb=False, name="mip", contrast_limits = (0, val))
            self.viewer.add_labels(self.list_masks[self.current_image_index], name= 'roi')
            shapes_layer = self.viewer.add_shapes(name='shapes', shape_type='polygon')
            shapes_layer.mode = 'add_polygon_lasso'          
        else:
            self.viewer.add_image(np.zeros(self.im_2d_dimensions, dtype=np.uint8), rgb=False, name="mip", contrast_limits = (0, val))
            self.viewer.add_labels(np.zeros(self.im_2d_dimensions, dtype=np.uint8), name= 'roi')
            shapes_layer = self.viewer.add_shapes(name='shapes', shape_type='polygon')
            shapes_layer.mode = 'add_polygon_lasso'       

        my_button = QPushButton(f"Image 0 {Path(self.list_im[0]).stem}: new image")
        my_button.setFixedSize(400, 30)
        my_button.clicked.connect(
            lambda: self.display_next_image(my_button)
        )
        self.viewer.window.add_dock_widget(my_button, area='bottom')

        my_button2 = QPushButton(f"Delete")
        my_button2.setFixedSize(400, 30)
        my_button2.clicked.connect(
            lambda: self.delete_mask_and_save()
        )
        self.viewer.window.add_dock_widget(my_button2, area='bottom')

        napari.run()
