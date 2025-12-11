import numpy as np
import napari
from napari.qt import QtViewerButtons
from pathlib import Path
from PIL import Image
from superqt import QLabeledSlider
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel
from pathlib import Path

from refine_seg import Segmentation
from utils.file_handling import FileProcessor 

fp = FileProcessor()
sg = Segmentation()


class ImageThresholding_v3:
    """
    Encapsulates the image thresholding functionality.

    Attributes:
        thresh_i_pixa: A list of tuples containing intensity and pixel size thresholds for each image.
        labels_to_rem: A list of lists containing labels to remove for each image (excluded cells labels).
        DIAMETER_DB: Diameter for deblurring.
        GPU: Whether to use GPU for processing.
        PRETRAINED_MODEL: Path to the pretrained model for cell segmentation.
        max_i_thresh: Maximum intensity threshold value.
        maw_px_thresh: Maximum pixel size threshold value.
        BATCH: A list of image paths.

    Methods:
        determine_masks_to_remove: Removes masks based on intensity and pixel size thresholds.
        intensity_in_mask_n: Calculates the intensity within a mask.
        mask_area: Calculates the area of a mask.
        recap_ints_and_areas: Calculates intensity and pixel size for each mask.
        threshold_image: Applies a simple threshold to the image.
        update_threshold: Updates the displayed image with the thresholded image.
        update_threshold_pix_area: Updates the displayed image with the thresholded image based on pixel area.
        change_image: Changes the displayed image and updates the threshold accordingly.
        __init__: Initializes the class with default parameters.
        run: Sets up the Napari viewer and starts the interactive thresholding process.
    """

    def __init__(
        self,
        BATCH,  # batch_1 should be defined externally
        PRETRAINED_MODEL: str,
        thresh_i_pixa=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        labels_to_rem=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        DIAMETER_DB=35,
        GPU=True,
        maw_px_thresh=1000,  # 200**2/32
        BATCH_NAME="",
        MASK_NUC_LIST_PATH= None,
        MASK_CONTOURS_LIST_PATH=None,
    ):
        """
        Initializes the ImageThresholding class with the given parameters.

        Args:
            thresh_i_pixa: See class docstring.
            labels_to_rem: See class docstring.
            DIAMETER_DB: See class docstring.
            GPU: See class docstring.
            PRETRAINED_MODEL: See class docstring.
            max_i_thresh: See class docstring.
            maw_px_thresh: See class docstring.
            BATCH: See class docstring.
        """
        self.thresh_i_pixa = thresh_i_pixa
        self.labels_to_rem = labels_to_rem
        self.DIAMETER_DB = DIAMETER_DB
        self.GPU = GPU
        self.PRETRAINED_MODEL = PRETRAINED_MODEL
        self.maw_px_thresh = int(maw_px_thresh)
        self.BATCH = BATCH
        self.current_image_index = 0
        self.batch_name = BATCH_NAME
        self.mask_nuc_list_path = MASK_NUC_LIST_PATH
        self.mask_contours_list = MASK_CONTOURS_LIST_PATH
        
        dict_thresh_i_pixa = {}
        batch_np = np.array(BATCH).flatten()

        name = str(np.array(self.BATCH).flatten()[self.current_image_index].stem)
        
        self.name_im = '_'.join(name.split('_')[:-3])
        
        
        for path in batch_np:
            name_temp = path.stem
            name_temp = '_'.join(name_temp.split('_')[:-3])
            
            dict_thresh_i_pixa[name_temp] = ([], [])

        self.dict_thresh_i_pixa = dict_thresh_i_pixa

        dict_labels_to_rem = {}
        for path in batch_np:
            name_temp = path.stem
            name_temp = '_'.join(name_temp.split('_')[:-3])    
            dict_labels_to_rem[name_temp] = []
        self.dict_labels_to_rem = dict_labels_to_rem

        im  = np.array(Image.open(np.array(self.BATCH).flatten()[self.current_image_index]))

        if im.ndim == 2:
            self.im_nuc = im
        elif im.ndim == 3:
            self.im_nuc = np.max(im, axis=0)
        else:
            raise ValueError("Image dimension not supported")

        self.max_i_thresh = np.max(self.im_nuc.flatten())
        self.list_im = np.array(self.BATCH).flatten()

    def determine_masks_to_remove(
        self,
        masks: np.ndarray,
        mean_i: list,
        pix_size: list,
        label_curr: list,
        int_threshold: int,
        px_threshold: int,
    ) -> np.ndarray:
        """Remove masks whose intensity is above a threshold and pixel size below another threshold.

        Args:
            masks (np.ndarray): each mask has a unique label, a integer.
            mean_i (list): Intensity scalar for each mask.
            pix_size (list): Area of each mask.
            label_curr (list): list of labels associated with the lists mean_i and pix_size.
            int_threshold (int): below this threshold of inensity in the mask, the mask is potentially removed.
            px_threshold (int): below this area of the mask, the mask is potentially removed.

        Returns:
            np.ndarray: masks, with the masks to remove set to 1, for visualization purposes.
        """
        masks_and = np.zeros_like(masks)
        ind_pix = np.logical_and(
            (np.array(pix_size) < px_threshold), (np.array(mean_i) > int_threshold)
        )
        labels_to_rem = np.array(label_curr)[ind_pix]
        for u in labels_to_rem:
            masks_and[masks == u] = 1
        return masks_and, labels_to_rem
    
    
    def determine_masks_to_remove_v2(
        self,
        masks: np.ndarray,
        mean_i: list,
        pix_size: list,
        label_curr: list,
        int_threshold: int,
        px_threshold: int,
    ) -> np.ndarray:
        """Remove masks whose area is below a threshold, and above the remaining masks,
        remove the masks whose intensity is above a threshold.
        
        Args:
            masks (np.ndarray): each mask has a unique label, a integer.
            mean_i (list): Intensity scalar for each mask.
            pix_size (list): Area of each mask.
            label_curr (list): list of labels associated with the lists mean_i and pix_size.
            int_threshold (int): below this threshold of inensity in the mask, the mask is potentially removed.
            px_threshold (int): below this area of the mask, the mask is potentially removed.

        Returns:
            np.ndarray: masks, with the masks to remove set to 1, for visualization purposes.
        """
        masks_and = np.zeros_like(masks)
        ind_pix1  = (np.array(pix_size) < px_threshold)
        ind_pix2  = np.logical_and(np.array(pix_size) >= px_threshold, 
                                   np.array(mean_i) > int_threshold)        
        ind_pix   = np.logical_or(ind_pix1, ind_pix2)

        labels_to_rem = np.array(label_curr)[ind_pix]
        for u in labels_to_rem:
            masks_and[masks == u] = 1
        return masks_and, labels_to_rem
        
    def intensity_in_mask_n(
        self, masks: np.ndarray, image: np.ndarray, lab_n: int
    ) -> float:
        """Characterize the intensity in a mask.
        Args:
            masks (np.ndarray): each mask has a unique label, a integer.
            image (np.ndarray): the image to be analyzed.
            lab_n (int): label number of the mask to be analyzed.

        Returns:
            float: the intensity value.
        """
        return np.percentile(image[masks == lab_n].flatten(), 95)

    def mask_area(self, masks: np.ndarray, lab_n: int) -> int:
        """Characterize the area of a mask.
        Args:
            masks (np.ndarray): each mask has a unique label, a integer.
            lab_n (int): label number of the mask to be analyzed.

        Returns:
            int: the area of the mask.
        """
        return np.sum((masks == lab_n) * 1)

    def recap_ints_and_areas(self, image: np.ndarray, masks: np.ndarray) -> tuple:
        """for each mask, compute the intensity and the pixel size.

        Args:
            image (np.ndarray): denoised image.
            masks (np.ndarray): each mask has a unique label, a integer.

        Returns:
            tuple: (mean_i, pix_size, label_curr): lists of intensity,
            pixel size and label number, for each label.
        """
        pix_size = []
        mean_i = []
        labels_u = np.unique(masks)
        label_curr = []
        for lab_n in labels_u:
            if lab_n:  # avoid the background (label 0)
                mean_i.append(self.intensity_in_mask_n(masks, image, lab_n))
                pix_size.append(self.mask_area(masks, lab_n))
                label_curr.append(lab_n)
        return mean_i, pix_size, label_curr

    def update_threshold(self, viewer, image, threshold1, threshold2):
        """
        Updates the displayed image in the napari viewer with the
        thresholded image.

        Args:
            viewer: The napari viewer instance.
            threshold1: The intensity threshold value.
            threshold2: The pixel area threshold value.
        """
        if (self.list_im[self.current_image_index] is not None) and (self.list_im[self.current_image_index] != Path('.')):
            row, col = np.unravel_index(self.current_image_index, (3, 3))
            self.thresh_i_pixa[row][col] = (threshold1, threshold2)
            name = str(np.array(self.BATCH).flatten()[self.current_image_index].stem)

            self.name_im =  '_'.join(name.split('_')[:-3])

            self.dict_thresh_i_pixa[self.name_im] = (threshold1, threshold2)
            viewer.layers[2].data, lab_to_rem = self.determine_masks_to_remove_v2(
                self.masks,
                self.mean_i,
                self.pix_size,
                self.label_curr,
                threshold1,
                threshold2,
            )
            self.labels_to_rem[row][col] = lab_to_rem
            self.dict_labels_to_rem[self.name_im] = lab_to_rem

    def change_image(self, viewer, button, my_slider, my_slider2):
        """
        Function to change the displayed image and update the threshold accordingly.

        Args:
            viewer: The napari viewer instance.
            current_image_index: The index of the currently displayed image.
            button: The button that triggered the change.
        """

        self.current_image_index = (self.current_image_index + 1) % len(self.list_im)
        
        if (self.list_im[self.current_image_index] is not None) and (self.list_im[self.current_image_index] != Path('.')):
        
            im = np.array(Image.open(self.list_im[self.current_image_index]))

            if im.ndim == 2:
                self.im_nuc = im
            elif im.ndim == 3:
                self.im_nuc = np.max(im, axis=0)
            else:
                raise ValueError("Image dimension not supported")
            self.max_i_thresh = np.max(self.im_nuc.flatten()) +1
            my_slider.setMaximum(self.max_i_thresh)
            name = str(np.array(self.BATCH).flatten()[self.current_image_index].stem)
            
            self.name_im =  '_'.join(name.split('_')[:-3])

            viewer.layers[0].data = self.im_nuc
            row, col = np.unravel_index(self.current_image_index, (3, 3))
            try:
                if self.mask_nuc_list_path is not None and self.mask_contours_list is not None:
                    mask_nuc_list_path = self.mask_nuc_list_path
                    masks_nuc      = fp.load_masks_distributed_files(mask_nuc_list_path)
                    masks_contours = np.load(self.mask_contours_list, allow_pickle=True)[()]
                                                                        
                    masks = masks_nuc[self.name_im]
                    contours = masks_contours[self.name_im]

            except:
                image_d = sg.deblur_cellpose(
                    self.im_nuc, diameter=self.DIAMETER_DB, gpu=self.GPU
                )
                masks = sg.segment_with_custom_model(
                    image_d, self.PRETRAINED_MODEL, gpu=self.GPU
                )
                contours = sg.find_all_contours(masks)

            self.masks = masks
            viewer.layers[1].data = contours
            self.mean_i, self.pix_size, self.label_curr = self.recap_ints_and_areas(
                self.im_nuc, self.masks
            )

            if isinstance(self.thresh_i_pixa[row][col], int):
                my_slider.setValue(int(self.max_i_thresh))
                my_slider2.setValue((int(self.maw_px_thresh/10)))
                self.update_threshold(
                    viewer, self.im_nuc, my_slider.value(), my_slider2.value()
                )
            else:
                my_slider.setValue(self.thresh_i_pixa[row][col][0])
                my_slider2.setValue(self.thresh_i_pixa[row][col][1])
                self.update_threshold(
                    viewer,
                    self.im_nuc,
                    self.thresh_i_pixa[row][col][0],
                    self.thresh_i_pixa[row][col][1],
                )
        else:
            viewer.layers[0].data = np.zeros((100,100))
            viewer.layers[1].data = []
            viewer.layers[2].data = np.zeros((100,100),dtype=int)

            
        button.setText(f"Image {self.current_image_index}")

    def run(self):
        """
        Sets up the Napari viewer and starts the interactive thresholding process.
        """
        viewer = napari.Viewer(title='Select the ROIs to remove')

        try:
            if self.mask_nuc_list_path is not None and self.mask_contours_list is not None:
                mask_nuc_list_path = self.mask_nuc_list_path
                masks_nuc      = fp.load_masks_distributed_files(mask_nuc_list_path)
                masks_contours = np.load(self.mask_contours_list, allow_pickle=True)[()]

                masks = masks_nuc[self.name_im]
                contours = masks_contours[self.name_im]
        except:
            image_d = sg.deblur_cellpose(
                self.im_nuc, diameter=self.DIAMETER_DB, gpu=self.GPU
            )
            masks = sg.segment_with_custom_model(
                image_d, self.PRETRAINED_MODEL, gpu=self.GPU
            )
            contours = sg.find_all_contours(masks)

        self.masks = masks
        self.mean_i, self.pix_size, self.label_curr = self.recap_ints_and_areas(
            self.im_nuc, self.masks
        )
        row, col = np.unravel_index(self.current_image_index, (3, 3))
        masks_and, lab_to_rem = self.determine_masks_to_remove_v2(
            self.masks, self.mean_i, self.pix_size, self.label_curr, 0, 0
        )
        self.labels_to_rem[row][col] = lab_to_rem
        self.dict_labels_to_rem[self.name_im] = lab_to_rem

        # set the layers
        viewer.add_image(self.im_nuc, rgb=False, name="DAPI")
        viewer.add_shapes(
            contours,
            name="Contours",
            shape_type="polygon",
            edge_color="red",
            face_color="transparent",
            opacity=1,
        )
        intensity_pix_layer = viewer.add_labels(
            masks_and,
            name="Intensity and area thresholded",
            opacity=0.5,
            blending="translucent",
        )
        intensity_pix_layer.color = {1: "blue"}

        button_widget = QtViewerButtons(viewer)
        my_button = QPushButton("Image 0: new image")
        my_button.setFixedSize(150, 30)

        # Create the slider Intensity
        my_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        my_slider.setMinimum(0)
        my_slider.setMaximum(self.max_i_thresh+1)
        my_slider.setValue(int(self.max_i_thresh))
        my_slider.setFixedWidth(800)
        my_slider.sliderReleased.connect(
            lambda: self.update_threshold(
                viewer, self.im_nuc, my_slider.value(), my_slider2.value()
            )
        )

        # Create the slider (pixel size)
        my_slider2 = QLabeledSlider(Qt.Orientation.Horizontal)
        my_slider2.setMinimum(0)
        my_slider2.setMaximum(int(self.maw_px_thresh)+1)
        my_slider2.setValue(int(self.maw_px_thresh/10))

        my_slider2.setFixedWidth(800)
        my_slider2.sliderReleased.connect(
            lambda: self.update_threshold(
                viewer, self.im_nuc, my_slider.value(), my_slider2.value()
            )
        )

        # button : change image
        my_button.clicked.connect(
            lambda: self.change_image(viewer, my_button, my_slider, my_slider2)
        )

        # Create a custom widget with a larger label
        widget = QWidget()
        layout = QVBoxLayout()
        title_label = QLabel("Intensity Threshold")
        title_label.setStyleSheet("font-size: 12pt")
        layout.addWidget(title_label)
        layout.addWidget(my_slider)
        widget.setLayout(layout)
        viewer.window.add_dock_widget(widget, area="bottom")

        # Create a QLabel for pixel area thresholding
        widget2 = QWidget()
        title_label = QLabel("Area Threshold")
        title_label.setStyleSheet("font-size: 12pt")
        layout.addWidget(title_label)
        layout.addWidget(my_slider2)
        widget2.setLayout(layout)
        viewer.window.add_dock_widget(widget2, area="bottom")

        butt_widget = QWidget()
        butt_layout = QVBoxLayout()
        butt_layout.addWidget(my_button)
        butt_widget.setLayout(butt_layout)

        viewer.window.add_dock_widget(butt_widget, area="bottom")

        napari.run()
