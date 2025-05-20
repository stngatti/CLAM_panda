from tiatoolbox.wsicore.wsireader import WSIReader

import numpy as np
import cv2
import os
import argparse
from file_utils import save_pkl
from PIL import Image

def generate_and_save_tiatoolbox_contours(wsi_path, output_pkl_path):
    """
    generates a .pkl file with contours from a WSI using Tiatoolbox segmentation.
    """

    print(f"Processing WSI: {wsi_path}")
    wsi = WSIReader.open(wsi_path)

    mask_resolution_val = 1.25
    mask_units_val = 'power'

    try:
        mask_reader_obj = wsi.tissue_mask(
            resolution=mask_resolution_val, 
            units=mask_units_val
        )

        mask_width, mask_height = mask_reader_obj.info.level_dimensions[0]

        raw_mask_data = mask_reader_obj.read_region(
            location=(0,0), 
            level=0, 
            size=(mask_width, mask_height)
        )
        
        if isinstance(raw_mask_data, Image.Image):
            print("INFO: raw_mask_data is a PIL Image. Converting into NumPy array.")
            binary_mask_np = np.array(raw_mask_data.convert('L'))
        elif isinstance(raw_mask_data, np.ndarray):

            binary_mask_np = raw_mask_data

            if binary_mask_np.ndim == 3:
                if binary_mask_np.shape[2] == 1: # (H, W, 1) check if the last dimension is 1
                    binary_mask_np = binary_mask_np.squeeze(axis=2)
                elif binary_mask_np.shape[2] >= 3: # (H, W, 3) o (H, W, 4) if RGB or RGBA
                    print(f"INFO: NumPy array has {binary_mask_np.shape[2]} channels. Converting to grayscale.")
                    binary_mask_np = binary_mask_np[:, :, 0] 
        else:
            msg = f"unexpected type of: {type(raw_mask_data)}"
            print(f"ERROR: {msg}")
            raise TypeError(msg)
        
        if binary_mask_np.dtype != np.uint8:
            if np.max(binary_mask_np) <= 1.0 and np.min(binary_mask_np) >=0: # es. se è float 0-1
                 binary_mask_np = (binary_mask_np * 255)
            binary_mask_np = binary_mask_np.astype(np.uint8)


        if np.max(binary_mask_np) == 1: #binary mask with tia is 1/0
            binary_mask_np = binary_mask_np * 255
        elif np.any((binary_mask_np != 0) & (binary_mask_np != 255)):
            print("WARNING: Mask is not binary (0/255). Converting to binary...")
            _, binary_mask_np = cv2.threshold(binary_mask_np, 0, 255, cv2.THRESH_BINARY)
        
        target_downsample = None
        try:
            #convert the mask resolution to baseline 
            target_downsample = wsi.convert_resolution_units(
                resolution=mask_resolution_val,
                units=mask_units_val,           
                new_units='baseline'
            )
            if isinstance(target_downsample, (tuple, list)):
                target_downsample = np.mean(target_downsample)

            print(f"INFO: Target downsample inferred from {mask_resolution_val} {mask_units_val}: {target_downsample}")

        except Exception as e:
            target_downsample = 32.0 

        if target_downsample is None: 
            target_downsample = 32.0

        level_downsamples_from_wsi = wsi.info.level_downsamples
        
        processed_level_downsamples = []
        for ds_val in level_downsamples_from_wsi:
            if isinstance(ds_val, (tuple, list)):
                processed_level_downsamples.append(np.mean(ds_val))
            else:
                processed_level_downsamples.append(float(ds_val))

        if not processed_level_downsamples:
            level_of_original_wsi_for_scaling = 0 
        else:
            differences = [abs(ds - target_downsample) for ds in processed_level_downsamples]
            level_of_original_wsi_for_scaling = np.argmin(differences)

    except Exception as e:
        print(f"Error while generating mask: {e}")
        import traceback
        traceback.print_exc()
        return

    #extract contours from the mask
    contours_on_mask, hierarchy_on_mask = cv2.findContours(binary_mask_np, 
                                                              cv2.RETR_CCOMP, 
                                                              cv2.CHAIN_APPROX_SIMPLE)

    # scale the contours to the original WSI level

    if not wsi.info or not hasattr(wsi.info, 'level_downsamples') or not wsi.info.level_downsamples:
        return 

    current_level_downsample = wsi.info.level_downsamples[level_of_original_wsi_for_scaling]
    if isinstance(current_level_downsample, (tuple, list)):
        scale_x = current_level_downsample[0]
        scale_y = current_level_downsample[1]
    else: 
        scale_x = current_level_downsample
        scale_y = current_level_downsample

    scaled_tissue_contours = [] #list of lists, each sublist contains the contours for the corresponding tissue contour
    temp_holes_for_tissue_contours = [] 

    if hierarchy_on_mask is not None and len(contours_on_mask) > 0:
        hierarchy_on_mask = np.squeeze(hierarchy_on_mask, axis=(0,)) # remove the first dimension if it is 1
        if hierarchy_on_mask.ndim == 1: 
            hierarchy_on_mask = np.array([hierarchy_on_mask])

        original_tissue_indices_map = {} 
        current_new_tissue_idx = 0
        for i, contour_at_lvl in enumerate(contours_on_mask):
            if hierarchy_on_mask[i][3] == -1: 
                scaled_contour = (contour_at_lvl * np.array([scale_x, scale_y])).astype(np.int32)
                scaled_tissue_contours.append(scaled_contour)
                original_tissue_indices_map[i] = current_new_tissue_idx
                temp_holes_for_tissue_contours.append([]) 
                current_new_tissue_idx += 1
        
        for i, contour_at_lvl in enumerate(contours_on_mask):
            parent_idx_original = hierarchy_on_mask[i][3]

            if parent_idx_original != -1 and parent_idx_original in original_tissue_indices_map:
                new_tissue_idx_for_hole = original_tissue_indices_map[parent_idx_original]
                scaled_hole = (contour_at_lvl * np.array([scale_x, scale_y])).astype(np.int32)
                temp_holes_for_tissue_contours[new_tissue_idx_for_hole].append(scaled_hole)
    
    scaled_hole_contours_by_tissue = temp_holes_for_tissue_contours

    asset_dict = {'tissue': scaled_tissue_contours, 'holes': scaled_hole_contours_by_tissue}
    
    output_dir = os.path.dirname(output_pkl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creata directory di output: {output_dir}")

    save_pkl(output_pkl_path, asset_dict)
    print(f"Contourns saved in: {output_pkl_path}")
    print(f"Number of saved contourns: {len(scaled_tissue_contours)}")
    num_holes_total = sum(len(holes) for holes in scaled_hole_contours_by_tissue)
    print(f"Number of holes: {num_holes_total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates contourns for ' )
    parser.add_argument('--wsi_path', type=str, required=True, help='Path to WSI file.')
    parser.add_argument('--output_pkl_path', type=str, required=True, help='Path to output .pkl file.')
    parser.add_argument('--mask_level', type=int, default=-1, help='WSI level for mask generation. Default -1 means use the best level for downsample 32.')

    args = parser.parse_args()

    tiatoolbox_custom_params = {}
    if args.mask_level != -1:
        tiatoolbox_custom_params['level'] = args.mask_level

    generate_and_save_tiatoolbox_contours(args.wsi_path, args.output_pkl_path)