# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#%%

import argparse
#from spots_detection import detection_folder_with_segmentation  # , detection_folder_without_segmentation
# from stiching import stich_with_image_J, parse_txt_file
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Press the green button in the gutter to run the script.

dico_bc_gene0 = {
    'r1_bc1': "Rtkn2",
    'r3_bc4': "Pecam1",
    'r4_bc5': "Ptprb",
    'r5_bc6': "Pdgfra",
    'r6_bc7': "Chil3",
    'r7_bc3': "Lamp3",
    'r8_bc3': "Rtkn2"

}

dico_bc_gene1 = {
    'r1': "Rtkn2",
    'r2': "Lamp3",
    'r3': "Pecam1",
    'r4': "Ptprb",
    'r5': "Pdgfra",
    'r6': "Chil3",
    'r7': "Apln",
    "r8": "Fibin",
    "r9": "Pon1",
    "r10": "Cyp2s1",
    "r11": "C3ar1",
    "r12": "Hhip",
    #"r13": "Hhip",
}



dico_bc_gene0 = {
    'r1': "Rtkn2", #bc1
    'r2': "Lamp3", # bc3
    'r3': "Pecam1", #bc4
    'r4': "Ptprb", #bc5
    'r5': "Pdgfra", #bc6
    'r6': "Chil3", #bc7
    'r7': "Rtkn2",  # bc7

    'r8': "Lamp3",  # bc7
    'r9': "Pecam1",  # bc7
    'r10': "Rtkn2",  # bc7

}


dico_bc_noise = {"r1_Cy5": "artefact"}


if __name__ == '__main__':

    ####### segment individual tile

    # input folder name + regex_file_name
    # result a new save folder with the same name + _segmented

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--path_NDTiffStack_to_tiff",
                        type=str,
                        default="/home/tom/Bureau/2023-10-06_LUSTRA_raw/",
                        help='')

    parser.add_argument("--folder_of_rounds",
                        type=str,
                        default="/home/tom/Bureau/2023-10-06_LUSTRA/",
                        help='')

    parser.add_argument("--path_to_round_for_segmentation",
                        type=str,
                        default="/home/tom/Bureau/2023-10-06_LUSTRA/r1",
                        help='')

    parser.add_argument("--path_to_mask_dapi",
                        type=str,
                        default="/home/tom/Bureau/2023-10-06_LUSTRA/segmentation_mask/",
                        help='')
    parser.add_argument("--regex_dapi",
                        type=str,
                        default="ch1",
                        help='')

    parser.add_argument("--fixed_round_name",
                        type=str,
                        default="r1",
                        help='')
    parser.add_argument("--folder_regex_round",
                        type=str,
                        default="[^r]*",
                        help='')
    parser.add_argument("--chanel_regex",
                        type=str,
                        default="*ch0*",
                        help='channel for the fish')

    parser.add_argument("--image_name_regex",
                        type=str,
                        default="[^r]*",
                        help='')

    parser.add_argument("--name_dico",
                        type=str,
                        default="26_oct",
                        help='')
    parser.add_argument("--scale_xy", type=float, default=0.108)
    parser.add_argument("--scale_z", type=float, default=0.300)

    ### param for detection
    parser.add_argument("--local_detection",
                        type=int,
                        default=0,
                        help='')

    parser.add_argument("--use_median_threshold",
                        type=int,
                        default=0,
                        help='')
    parser.add_argument("--mask_artefact", type=int, default=0)
    parser.add_argument("--artefact_filter_size", type=int, default=32)
    parser.add_argument("--sigma_detection", type=float, default=1.3)
    parser.add_argument("--remove_non_sym", type=int, default=0)

    ######## parameters stiching
    parser.add_argument("--image_shape",
                        type=list,
                        default=[38, 2048, 2048],
                        help='')

    parser.add_argument("--nb_tiles",
                        type=int,
                        default=5,
                        help='')

    parser.add_argument("--regex_image_stiching",
                        type=str,
                        default="r1_pos{i}_ch0.tif",
                        )

    parser.add_argument("--remove_double_detection", type = int, default = 0)
    parser.add_argument("--stich_segmentation_mask", type = int, default = 1)

    ##### task to do
    parser.add_argument("--NDTiffStack_to_tiff", default=1, type=int)

    parser.add_argument("--segmentation", default=0, type=int)
    parser.add_argument("--registration", default=1, type=int)

    parser.add_argument("--spots_detection", default=1, type=int)
    parser.add_argument("--signal_quality", default=0, type=int)
    parser.add_argument("--spots_registration", default=1, type=int)

    parser.add_argument("--stitch", default=0, type=int)
    parser.add_argument("--stich_spots_detection", default=0, type=int)
    parser.add_argument("--plot_control_quality", default=0, type=int)
    parser.add_argument("--generate_comseg_input", default=0, type=int)

    parser.add_argument("--port", default=39948)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--host", default='127.0.0.2')

    args = parser.parse_args()

    print(args)


    args.image_path_stiching = args.folder_of_rounds + args.fixed_round_name
    args.output_path_stiching = args.folder_of_rounds + args.fixed_round_name + "output_s"
    Path(args.image_path_stiching).mkdir(exist_ok=True, parents=True)
    Path(args.output_path_stiching).mkdir(exist_ok=True, parents=True)


    e = datetime.datetime.now()
    date_str = f"{e.month}_{e.day}_{e.hour}_{e.minute}_{e.second}"
    #%%

    ####################
    ####### segment individual tile
    ####################


    ## NDTiffStack to img folder

    if args.NDTiffStack_to_tiff == 1:

        from autofish_analysis.split_ndtiff_stack import NDTiffStack_to_tiff

        NDTiffStack_to_tiff(
            path_parent=args.path_NDTiffStack_to_tiff + 'test/',
            n_z=41,  # number of z plances
            n_pos=9,  # number of field of images

            # Number of channels
            n_c_default=1,
            n_c_dict={'r1_1': 2}, ## round with the dapi

            # Diverse string replacements
            string_replacements_path=[('images_multi-stack', 'images'),
                                      ('_1', '')],
            string_replacements_file=[('_NDTiffStack.*', ''),
                                      ("_bgd*", ""),
                                      ('Pos', 'pos'),
                                      ],
            folder_save=args.folder_of_rounds
        )

    if args.segmentation == 1:
        from pathlib import Path

        from autofish_analysis.segmentation import folder_segmentation
        print("segmentation")
        ### set the cellpose parameters
        dico_param = {}
        dico_param["diameter"] = 50
        dico_param["flow_threshold"] = 0.6
        dico_param["mask_threshold"] = 0
        dico_param["do_3D"] = False
        dico_param["mip"] = False
        dico_param["projected_focused"] = False
        dico_param["stitch_threshold"] = 0.3
        dico_param["erase_solitary"] = True
        dico_param["erase_small_nuclei"] = 300
        dico_param["model_type"] = "cyto"
        dico_param["gpu"] = True
        folder_segmentation(path_to_staining=args.path_to_round_for_segmentation,
                            regex_staining=args.regex_dapi,
                            path_to_mask=args.path_to_mask_dapi + '50/',
                            dico_param=dico_param,
                            output_dtype=np.int32
                            )

    ########
    # register each channel to the dapi round channel
    #######

    if args.registration == 1:


        from autofish_analysis.registration import folder_translation

        dico_translation = folder_translation(folder_of_rounds=args.folder_of_rounds,  # works ok
                                              fixed_round_name=args.fixed_round_name,
                                              folder_regex=args.folder_regex_round,
                                              chanel_regex=args.chanel_regex,
                                              registration_repeat=5,
                                              sigma_gaussian_filter = 0.9,
                                              max_rotation_accepted=0.05,
                                              )

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy", dico_translation)

    if args.plot_registration:
        from autofish_analysis.registration import plot_registrered_image
        dico_translation =  np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy", allow_pickle=True).item()

        list_round = list(dico_translation[list(dico_translation.keys())[0]][args.fixed_round_name].keys())
        list_pos = list(dico_translation.keys())
        for rd in list_round:
            if rd == args.fixed_round_name:
                continue
            path_save = Path(args.folder_of_rounds) / (rd + "/" + "registration")
            path_save.mkdir(exist_ok=True, parents=True)

            for pos in list_pos:
                path_image1= Path(args.folder_of_rounds) / (args.fixed_round_name + '/' + args.fixed_round_name + "_" + pos + "_ch0.tif")
                path_image2= Path(args.folder_of_rounds) / (rd + '/' + rd + "_" + pos + "_ch0.tif")



                fig, ax = plot_registrered_image(
                    dico_translation,
                    path_image1=path_image1,
                    path_image2=path_image2,
                    plot_napari=False,
                    figsize=(15, 15)
                )
                fig.savefig(path_save / Path(path_image2).stem )



    ####################
    ## Do individual spots detection (optional : using mask segmentation)
    ####################
    if args.spots_detection == 1:
        from autofish_analysis.spots_detection import folder_detection
        print("spots detection")
        if args.local_detection:
            dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                       allow_pickle=True).item()
        else:
            dico_translation = None

        if args.mask_artefact:
            #dico_spot_artefact = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spot_artefact.npy",
             #                          allow_pickle=True).item()
            dico_spot_artefact = np.load("/home/tom/Bureau/2023-10-06_LUSTRA/26_oct_dico_spot_artefact_r1_noise_r6.npy", allow_pickle=True).item()
        else:
            dico_spot_artefact = None
            artefact_filter_size = None
        threshold_input = None
        dico_spots, dico_threshold = folder_detection(
            folder_of_rounds=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            fixed_round_name=args.fixed_round_name,
            path_output_segmentaton=args.path_to_mask_dapi,
            min_distance=(4, 4, 4),
            scale_xy=args.scale_xy,
            scale_z=args.scale_z,
            sigma=args.sigma_detection,
            ## mask artefact
            dico_spot_artefact=dico_spot_artefact,
            artefact_filter_size = args.artefact_filter_size,
            ### detection parameters with segmentation
            dico_translation=dico_translation,
            diam_um=20,
            local_detection=args.local_detection,
            min_cos_tetha=0.70,
            order=5,
            test_mode=False,
            threshold_merge_limit=0.330,
            file_extension='tif',
            threshold_input=threshold_input,
                use_median_threshold = args.use_median_threshold,
            remove_non_sym=args.remove_non_sym,
        )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}" + \
                f"_{args.folder_regex_round}_mask_artefact{args.mask_artefact}_{args.sigma_detection}_remove_non_sym{args.remove_non_sym}.npy",
                dico_spots)

    if args.plot_spots_detection:





    ########
    # Compute signal quality for each round
    #######

    if args.signal_quality == 1:

        from utils.signal_quality import compute_quality_all_rounds
        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}_{args.folder_regex_round}.npy",
             allow_pickle=True).item()
        ######################" add the automatic plot
        dico_signal_quality = compute_quality_all_rounds(
            dico_spots = dico_spots,
            round_folder_path=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            file_extension="tif",
            voxel_size=[args.scale_z, args.scale_xy, args.scale_xy],
            spot_radius=900,
            sigma=1.3,
            order=5,
            compute_sym=True,
            return_list=True,
        )


        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_signal_quality{args.local_detection}.npy", dico_signal_quality)
        mean_intensity = []
        mean_snr = []
        mean_symmetry_coef = []
        mean_background = []
        for round_str in dico_signal_quality:


            median_intensity = []
            median_snr = []
            median_symmetry_coef = []
            median_background = []

            for image in dico_signal_quality[round_str]:
                mean_intensity += dico_signal_quality[round_str][image]["intensity"]
                mean_background += dico_signal_quality[round_str][image]["background"]
                mean_snr += dico_signal_quality[round_str][image]["snr"]
                mean_symmetry_coef += dico_signal_quality[round_str][image]["symmetry_coef"]

            median_intensity.append(np.median(mean_intensity))
            median_background.append(np.median(mean_background))
            median_snr.append(np.median(mean_snr))
            median_symmetry_coef.append(np.median(mean_symmetry_coef))
    #### create dataframe
    pd.DataFrame({"round":list(dico_signal_quality.keys()), "median_intensity":median_intensity, "median_background":median_background, "median_snr":median_snr,
                  "median_symmetry_coef":median_symmetry_coef}).to_csv(f"{args.folder_of_rounds}{args.name_dico}_signal_quality.csv", index=False)

    ########################### Registration of the spots ################
    if args.spots_registration == 1:
        from autofish_analysis.registration import spots_registration
        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}" + \
             f"_{args.folder_regex_round}_mask_artefact{args.mask_artefact}_{args.sigma_detection}_remove_non_sym{args.remove_non_sym}.npy",
             allow_pickle=True).item()
        dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                   allow_pickle=True).item()
        dico_spots_registered_df, dico_spots_registered, missing_data = spots_registration(
            dico_spots,
            dico_translation,
            fixed_round_name=args.fixed_round_name,
            check_removal=False,
            threshold_merge_limit=0.330,
            scale_xy=0.103,
            scale_z=0.270,
        )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}" , dico_spots_registered_df)
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_{args.fixed_round_name}" , dico_spots_registered)

    if args.dict_register_artefact:
        from autofish_analysis.stitching import dico_register_artefact

        dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                   allow_pickle=True).item()

        dico_spots_registered = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_{args.fixed_round_name}" + ".npy",
                                        allow_pickle=True).item()


        dico_spot_artefact = dico_register_artefact(
                                dico_spots_registered,
                               dico_translation,
                               dico_bc_noise={'r6': 'artefact'},
                               ref_round='r1',
                               list_round=['r2', 'r8', 'r1', 'r9', 'r4', 'r10', 'r5', 'r3', 'r7', 'r6']
                               )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spot_artefact_{args.fixed_round_name}_noise_r6" , dico_spots_registered)

    ########
    # Stitch only ref round images
    #######


    ##############
    #  add nucleus label or cell label to the spots using the non stiched label
    #############

    if args.add_label_to_spots == 1:
        from registration import spots_assignement
        dico_spots_registered_df = spots_assignement(dico_spots_registered_df,
                          path_to_masks=args.path_to_masks,
                          files_mask_extension="*.tif*", )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}" , dico_spots_registered_df)

    ########
    # Stitching
    #######

    if args.stitch == 1:
        ###################################
        ### compute the translation between the images
        ###################################
        import imagej
        import scyjava
        from  autofish_analysis.utils import macro
        #from utils.macro import STITCHING_MACRO

        scyjava.config.add_option('-Xmx40g')
        if "ij" not in vars():
            ij = imagej.init('sc.fiji:fiji')

        ### define the images to stitch and their position
        img_pos_dico = {"img0": ['pos0', "pos1", "pos2", 'pos3', 'pos4',
                                 'pos5', "pos6", "pos7", 'pos8', 'pos8'],
                       }

        import importlib

        from autofish_analysis import stitching

        importlib.reload(stitching)
        from autofish_analysis.stitching import stich_with_image_J

        stich_with_image_J(
            ij,
            STITCHING_MACRO=macro.STITCHING_MACRO,
            img_pos_dico=img_pos_dico,
            stitching_type="[Grid: snake-by-row]",
            order="[Left & Down]",
            grid_size_x=3,
            grid_size_y=3,
            tile_overlap=20,
            image_name="r1_pos{i}_ch0.tif",
            image_path=args.image_path_stiching,
            output_path=args.output_path_stiching)

        from autofish_analysis.stitching import parse_txt_file

        dico_centroid = {}
        for img_name in img_pos_dico:
            path_txt = Path(args.image_path_stiching) / f"TileConfiguration_{img_name}.txt"
            dico_stitch = parse_txt_file \
                (path_txt=path_txt,
                 image_name_regex="_pos", )
            dico_centroid[img_name] = dico_stitch
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dict_stitch_img.npy", dico_centroid)

        ###################################
        ############ point stitching
        ###################################
        from stitching import stich_dico_spots

        dico_spots_registered = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_{args.fixed_round_name}.npy",
                                        allow_pickle=True).item()
        dico_spots_registered_df =  np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}.npy",
                                        allow_pickle=True).item()
        dict_stitch_img = np.load(f"{args.folder_of_rounds}{args.name_dico}_dict_stitch_img.npy",
                                  allow_pickle=True).item()
        ######
        ## define the correspondance between round and RNA species
        ########

        dico_bc_gene = {
            'r1': "gene1",
            'r5': "gene2",
        },

        dico_spots_registered_stitch_df = stich_dico_spots(
            dico_spots_registered=dico_spots_registered,
            dico_stitch_img=dict_stitch_img,
            dico_bc_gene=dico_bc_gene,
            image_shape=args.image_shape,
            nb_tiles_x=3,
            nb_tiles_y=1,
        )

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_stitch_df.npy",
                                  allow_pickle=True).item()


        ########
        # image stitching seg mask
        ########
        from stitching import stich_segmask

        path_mask = "/media/tom/Transcend/autofish_test_stiching/segmentation_mask"
        path_to_save_mask = "/media/tom/Transcend/autofish_test_stiching/segmentation_mask_stitch"

        _, dico_centroid, df_matching_new_cell_label = stich_segmask(dict_stitch_img,
                                                         # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                                         path_mask=path_mask,
                                                         path_to_save_mask=path_to_save_mask,
                                                         image_shape=args.image_shape,
                                                         nb_tiles_x=nb_tiles_x,
                                                         nb_tiles_y=nb_tiles_y,
                                                         compute_dico_centroid=True,
                                                         iou_threshold=0.10)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_centroid.npy", dico_centroid)
        #np.save(f"{args.folder_of_rounds}{args.name_dico}_df_matching_new_cell_label.npy", df_matching_new_cell_label)
        df_matching_new_cell_label.to_csv(f"{args.folder_of_rounds}{args.name_dico}_df_matching_new_cell_label.csv")


        ######### fish signal stiching
        from stitching import stich_from_dico_img

        image_shape = [38, 2048, 2048]
        nb_tiles_x = 3
        nb_tiles_y = 1
        for image_name in dict_stitch_img:
            dico_stitch = dict_stitch_img[image_name]
            final_masks = stich_from_dico_img(dico_stitch,
                                              # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                              path_mask="/media/tom/Transcend/autofish_test_stiching/r1",
                                              regex="*_ch0*tif*",
                                              image_shape=image_shape,
                                              nb_tiles_x=nb_tiles_x,
                                              nb_tiles_y=nb_tiles_y, )

            np.save(
                Path("/media/tom/Transcend/autofish_test_stiching/segmentation_mask_stitch/stitch_input") / image_name,
                final_masks)


    ####################
    ### Add segmentnation label
    #####################
        if args.stitch:
            dict_spots_registered_df = dico_spots_registered_stitch_df


        dict_spots_registered_df = spots_assignement(dico_spots_registered_df,
                                                     path_to_masks=path_to_masks,
                                                     files_mask_extension="*.tif*",
                                                     in_place=False, )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_with_cell_df.npy", dict_spots_registered_df)
                ########
                # Genrate comseg imput
                #######

        if args.generate_andata == 1:
            from utils.post_processing_anndata import get_anndata
            anndata = get_anndata(dico_gene=dico_bc_gene,
                                  dico_spots_registered_df=dico_spots_registered_df,
                                  cell_column_name="cell_assignment",
                                  add_mask_polygone_cell=True,
                                  path_to_mask_cell="/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/segmentation_mask",
                                  )