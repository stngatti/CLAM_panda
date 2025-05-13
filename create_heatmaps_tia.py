from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from types import SimpleNamespace
import h5py
import yaml
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from PIL import Image 
from scipy.stats import percentileofscore 

from dataset_modules.wsi_dataset import Wsi_Region
from wsi_core.WholeSlideImageTia import WholeSlideImageTia
from wsi_core.wsi_utils import to_percentiles # Make sure it's imported if used in visHeatmap

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions modified for TIAToolbox
def initialize_wsi_for_tia(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    """
    Initializes a WholeSlideImageTia object, performs tissue segmentation using a fixed power resolution,
    and saves the segmentation.
    """
    wsi_object = WholeSlideImageTia(wsi_path)
    
    # Determine mask_resolution from seg_params
    mask_resolution_arg = {'resolution': 1.25, 'units': 'power'} 
    # print(f"Segmentazione forzata con risoluzione: {mask_resolution_arg}")

    if seg_params is None:
        seg_params = {}

    wsi_object.segmentTissue(mask_resolution=mask_resolution_arg, 
                             filter_params=filter_params,
                             keep_ids=seg_params.get('keep_ids', []),
                             exclude_ids=seg_params.get('exclude_ids', []))
    if seg_mask_path:
        wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def drawHeatmap_for_tia(scores, coords, slide_path=None, wsi_object=None, vis_level = 0, **kwargs):
    """
    Draws a heatmap on the WSI image using WholeSlideImageTia.
    """
    if wsi_object is None:
        wsi_object = WholeSlideImageTia(slide_path)
        # print(wsi_object.name) # Already present in WholeSlideImageTia.visHeatmap
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    
    top_left = wsi_kwargs.get('top_left') # Added .get for safety
    bot_right = wsi_kwargs.get('bot_right')
    patch_size_params = wsi_kwargs.get('patch_size') # patch_size is a tuple (w,h)
    
    # Wsi_Region should be adapted if necessary to use WholeSlideImageTia
    # and its reading methods (e.g., self.wsi_object.wsi.read_rect)
    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    # print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    # print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader, desc="Computing from Patches")):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.inference_mode():
            if roi.dim() == 3:
                roi = roi.unsqueeze(0) # Trasform [C, H, W] to [1, C, H, W]
            
            features = feature_extractor(roi)

            if attn_save_path is not None and model is not None: # Added check for model
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(0, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores) # score2percentile must be defined

                asset_dict = {'attention_scores': A, 'coords': coords}
                attn_save_path = save_hdf5(attn_save_path, asset_dict, mode=mode) # save_hdf5 returns the path
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            feat_save_path = save_hdf5(feat_save_path, asset_dict, mode=mode) # save_hdf5 returns the path

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            # model_results_dict = model(features) # Removed if not used
            logits, Y_prob, Y_hat, A, _ = model(features) # Added _ for the fifth output if present
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError(f"Model type {type(model)} not supported for inference.")
        # print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy() # Assuming the last element is of interest
        ids = ids[-1].cpu().numpy()     # Assuming the last element is of interest
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            try:
                # Handle the case where val might be an empty string for numeric types
                if isinstance(params[key], (int, float)) and isinstance(val, str) and val == '':
                    continue # Keep the default value if the string is empty
                
                val = dtype(val)
                if isinstance(val, str):
                    if len(val) > 0: # Do not overwrite with an empty string if not intended
                        params[key] = val
                elif not np.isnan(val): # Check NaN only for floats, not for strings
                    params[key] = val
            except ValueError:
                # print(f"Warning: Could not convert value '{val}' for key '{key}' to type {dtype}. Using default.")
                pass # Keep the default value if conversion fails
            # else: # Removed pdb.set_trace()
                # pass
    return params

def parse_config_dict(current_args, config_dict_from_yaml):
    if current_args.save_exp_code is not None:
        config_dict_from_yaml['exp_arguments']['save_exp_code'] = current_args.save_exp_code
    if current_args.overlap is not None:
        config_dict_from_yaml['patching_arguments']['overlap'] = current_args.overlap
    return config_dict_from_yaml

if __name__ == '__main__':
    config_path = args.config_file
    config_dict_yaml = yaml.safe_load(open(config_path, 'r'))
    config_dict_parsed = parse_config_dict(args, config_dict_yaml)

    args_all = config_dict_parsed # Renaming for clarity
    patch_args = argparse.Namespace(**args_all['patching_arguments'])
    data_args = argparse.Namespace(**args_all['data_arguments'])
    model_args_dict = args_all['model_arguments'] # Keep it as a dictionary for update
    model_args_dict.update({'n_classes': args_all['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args_dict)
    encoder_args = argparse.Namespace(**args_all['encoder_arguments'])
    exp_args = argparse.Namespace(**args_all['exp_arguments'])
    heatmap_args = argparse.Namespace(**args_all['heatmap_arguments'])
    if 'sample_arguments' in args_all and args_all['sample_arguments'] is not None:
        sample_args = argparse.Namespace(**args_all['sample_arguments'])
    else:
        sample_args = argparse.Namespace(samples=[]) #define default arguments
    
    patch_size_val = patch_args.patch_size # it's an int
    patch_size = tuple([patch_size_val, patch_size_val]) # convert to tuple
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    # print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset 
    
    def_filter_params = {'a_t':20.0, 'a_h': 4.0, 'max_n_holes':10}

    if preset is not None:
        try:
            preset_df = pd.read_csv(preset)
            for key in def_filter_params.keys():
                if key in preset_df.columns: def_filter_params[key] = preset_df.loc[0, key]
        except FileNotFoundError:
            print(f"Warning: Preset file {preset} not found. Default files will be used.")
        except Exception as e:
            print(f"Warning: Error while reading {preset}: {e}.")


    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides_fnames = []
            for data_dir_item in data_args.data_dir:
                slides_fnames.extend(os.listdir(data_dir_item))
        else:
            slides_fnames = sorted(os.listdir(data_args.data_dir))
        
        slides_fnames = [s for s in slides_fnames if data_args.slide_ext in s]
        df = pd.DataFrame({'slide_id': [s.replace(data_args.slide_ext, '') for s in slides_fnames]})
        df['process'] = 1 
    else:
        try:
            df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
            if 'process' not in df.columns:
                df['process'] = 1
        except FileNotFoundError:
            print(f"Errore: File process_list '{os.path.join('heatmaps/process_lists', data_args.process_list)}' non trovato.")
            exit()
        except Exception as e:
            print(f"Errore durante la lettura del process_list: {e}")
            exit()

    if 'isup_grade' in df.columns and 'label' not in df.columns:
        df['label'] = df['isup_grade']
    elif 'label' not in df.columns:
        df['label'] = 'Unspecified' # Fallback

    mask_process = df['process'] == 1
    process_stack = df[mask_process].reset_index(drop=True)
    total = len(process_stack)
    # print('\nlist of slides to process: ')
    # print(process_stack.head(total))

    # print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    # print('\nckpt path: {}'.format(ckpt_path))
    
    if model_args.initiate_fn == 'initiate_model':
        model =  initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    if feature_extractor is not None:
        _ = feature_extractor.eval()
        feature_extractor = feature_extractor.to(device)
    # print('Done!')

    label_dict =  data_args.label_dict if hasattr(data_args, 'label_dict') else {} 
    if not label_dict: 
        unique_labels = process_stack['label'].unique()
        if all(isinstance(x, (int, np.integer)) for x in unique_labels):
             label_dict = {i: str(i) for i in sorted(unique_labels) if i != 'Unspecified'}
        else: 
             label_dict = {label: label for label in unique_labels if label != 'Unspecified'}


    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values()) 
    if all(isinstance(k, int) for k in label_dict.keys()) and all(isinstance(v, str) for v in label_dict.values()):
        reverse_label_dict = label_dict
    else: 
        if all(isinstance(v, (int, np.integer)) for v in label_dict.values()): # Se i valori sono int (es. {"A":0, "B":1})
            reverse_label_dict = {v: k for k, v in label_dict.items()}
        else: 
            unique_df_labels = process_stack['label'].unique()
            reverse_label_dict = {l: str(l) for l in unique_df_labels}

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    
    blocky_wsi_kwargs = {
        'top_left': None, 'bot_right': None, 
        'patch_size': patch_size, 
        'step_size': patch_size, 
        'custom_downsample': patch_args.custom_downsample, 
        'level': patch_args.patch_level,                   
        'use_center_shift': heatmap_args.use_center_shift  
    }

    for i in tqdm(range(total), desc="Processing Slides"):
        slide_name_from_df = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name_from_df:
            slide_name = slide_name_from_df + data_args.slide_ext
        else:
            slide_name = slide_name_from_df
        # print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str) and label in reverse_label_dict:
            grouping = reverse_label_dict[label]
        else:
            grouping = str(label)

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1 = process_stack.loc[i].get('x1', None) 
            x2 = process_stack.loc[i].get('x2', None)
            y1 = process_stack.loc[i].get('y1', None)
            y2 = process_stack.loc[i].get('y2', None)
            if all(v is not None for v in [x1, x2, y1, y2]):
                top_left = (int(x1), int(y1))
                bot_right = (int(x2), int(y2))
            else:
                top_left = None
                bot_right = None
                if heatmap_args.use_roi:
                    print(f"Attenzione: use_roi è True ma le coordinate ROI non sono definite per {slide_id}. Verrà processata l'intera slide.")
        else:
            top_left = None
            bot_right = None
        
        # print('slide id: ', slide_id)
        # print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, list):
            found_slide = False
            for data_dir_item in data_args.data_dir:
                potential_path = os.path.join(data_dir_item, slide_name)
                if os.path.exists(potential_path):
                    slide_path = potential_path
                    found_slide = True
                    break
            if not found_slide:
                print(f"Slide {slide_name} not found in any specified data_dir.")
                continue
        elif isinstance(data_args.data_dir, dict):
            data_dir_key_col = data_args.data_dir_key 
            if data_dir_key_col in process_stack.columns:
                data_dir_key_val = process_stack.loc[i, data_dir_key_col]
                slide_path = os.path.join(data_args.data_dir[data_dir_key_val], slide_name)
            else:
                print(f"Errore: data_dir_key '{data_dir_key_col}' non trovato nel DataFrame per {slide_id}.")
                continue
        else:
            raise NotImplementedError("data_args.data_dir deve essere una stringa, lista o dizionario.")


        mask_file_path = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        current_filter_params = def_filter_params.copy()
        # Sovrascrivi filter_params dal CSV se presenti
        current_filter_params = load_params(process_stack.loc[i], current_filter_params) 

        keep_ids_val_str = str(process_stack.loc[i].get('keep_ids', 'none'))
        exclude_ids_val_str = str(process_stack.loc[i].get('exclude_ids', 'none'))

        seg_params_for_init = {
            'keep_ids': np.array(keep_ids_val_str.split(',')).astype(int) if len(keep_ids_val_str) > 0 and keep_ids_val_str != 'none' else [],
            'exclude_ids': np.array(exclude_ids_val_str.split(',')).astype(int) if len(exclude_ids_val_str) > 0 and exclude_ids_val_str != 'none' else []
        }
        
        # print('Initializing WSI object')
        wsi_object = initialize_wsi_for_tia(slide_path, seg_mask_path=mask_file_path, 
                                       seg_params=seg_params_for_init, 
                                       filter_params=current_filter_params)
        # print('Done initializing WSI object!')

        if patch_args.patch_level < len(wsi_object.level_downsamples):
            wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        else:
            wsi_ref_downsample = (1.0, 1.0) 

        vis_patch_size_for_heatmap = patch_size 

        block_map_h5_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_jpg_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        
        if not os.path.isfile(mask_jpg_path):
            vis_level_for_mask = int(process_stack.loc[i].get('vis_level_mask', heatmap_args.vis_level if heatmap_args.vis_level >=0 else 0))
            line_thickness_for_mask = int(process_stack.loc[i].get('line_thickness_mask', 250))
            
            vis_params_for_mask = {
                'vis_level': vis_level_for_mask,
                'line_thickness': line_thickness_for_mask,
                'custom_downsample': heatmap_args.custom_downsample
            }
            mask_img = wsi_object.visWSI(**vis_params_for_mask, number_contours=True)
            mask_img.save(mask_jpg_path)
        
        features_pt_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        features_h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5') 

        if not os.path.isfile(features_pt_path):
            if not os.path.isfile(features_h5_path):
                # print(f"H5 file not found at {features_h5_path}, computing features...")
                _, features_h5_path, wsi_object = compute_from_patches(
                    wsi_object=wsi_object, 
                    model=model, 
                    feature_extractor=feature_extractor, 
                    img_transforms=img_transforms,
                    batch_size=exp_args.batch_size, 
                    **blocky_wsi_kwargs,
                    attn_save_path=None, 
                    feat_save_path=features_h5_path
                )
            
            if features_h5_path and os.path.isfile(features_h5_path):
                with h5py.File(features_h5_path, "r") as file:
                    features = torch.tensor(file['features'][:])
                torch.save(features, features_pt_path)
            else:
                # print(f"Error: Could not load or compute features from H5 file: {features_h5_path}")
                continue # Skip to the next slide
        
        features = torch.load(features_pt_path)
        process_stack.loc[i, 'bag_size'] = len(features)

        Y_hats, Y_hats_str, Y_probs, A_raw = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
        del features # Free memory
        
        # Save A_raw and corresponding coordinates in block_map_h5_path
        if not os.path.isfile(block_map_h5_path): 
            # Coordinates for A_raw should come from feature extraction
            # If features_h5_path was created by compute_from_patches, it should contain coordinates
            if features_h5_path and os.path.isfile(features_h5_path):
                with h5py.File(features_h5_path, "r") as file:
                    if 'coords' in file:
                        coords_for_A = file['coords'][:]
                        asset_dict = {'attention_scores': A_raw, 'coords': coords_for_A}
                        block_map_h5_path = save_hdf5(block_map_h5_path, asset_dict, mode='w')
                    else:
                        print(f"Warning: 'coords' not found in {features_h5_path}. Cannot save blockmap.h5.")
            else:
                 print(f"Warning: Feature H5 file {features_h5_path} not found. Cannot save blockmap.h5.")

        for c_idx in range(exp_args.n_classes): # Save predictions
            process_stack.loc[i, f'Pred_{c_idx}'] = Y_hats_str[c_idx] if c_idx < len(Y_hats_str) else 'N/A'
            process_stack.loc[i, f'p_{c_idx}'] = Y_probs[c_idx] if c_idx < len(Y_probs) else float('nan')

        os.makedirs('heatmaps/results/', exist_ok=True)
        csv_save_path = os.path.join('heatmaps/results/', f"{exp_args.save_exp_code}.csv" if data_args.process_list is None else data_args.process_list.replace('.csv', '_results.csv'))
        process_stack.to_csv(csv_save_path, index=False)
        
        # Load scores and coords for the heatmap from block_map_h5_path
        if block_map_h5_path and os.path.isfile(block_map_h5_path):
            with h5py.File(block_map_h5_path, 'r') as file:
                scores_for_heatmap = file['attention_scores'][:]
                coords_for_heatmap = file['coords'][:]
        else:
            # print(f"Blockmap H5 file not found: {block_map_h5_path}. Skipping heatmap generation.")
            continue

        # Patch sampling (if enabled)
        samples_config = sample_args.samples
        if isinstance(samples_config, list): # Ensure it's a list
            for sample_entry in samples_config:
                if sample_entry.get('sample', False): # Use .get for safety
                    tag = f"label_{label}_pred_{Y_hats[0]}"
                    sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample_entry['name'])
                    os.makedirs(sample_save_dir, exist_ok=True)
                    # print(f"Sampling for: {sample_entry['name']}")
                    sample_results = sample_rois(scores_for_heatmap, coords_for_heatmap, 
                                                 k=sample_entry.get('k', 5), 
                                                 mode=sample_entry.get('mode', 'top'), 
                                                 seed=sample_entry.get('seed', 123), 
                                                 score_start=sample_entry.get('score_start', 0), 
                                                 score_end=sample_entry.get('score_end', 1))
                    for patch_idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                        # print(f'coord: {s_coord} score: {s_score:.3f}')
                        # s_coord are at level 0. patch_args.patch_level is the level at which patch_args.patch_size is defined.
                        # The patch size to read is patch_args.patch_size at patch_args.patch_level.
                        try:
                            patch_img = wsi_object.wsi.read_rect(
                                location=tuple(s_coord), # Coordinates at level 0
                                size=(patch_args.patch_size, patch_args.patch_size), # Desired patch size
                                resolution=patch_args.patch_level, # Level to which patch_size refers
                                units='level'
                            ).convert('RGB')
                            patch_img.save(os.path.join(sample_save_dir, f'{patch_idx}_{slide_id}_x_{s_coord[0]}_y_{s_coord[1]}_a_{s_score:.3f}.png'))
                        except Exception as e:
                            print(f"Error reading/saving sampled patch at {s_coord}: {e}")


        # Arguments for compute_from_patches when calculating the final heatmap (if use_roi)
        # wsi_kwargs for the final heatmap (if recalculating attention scores with ROI)
        final_heatmap_wsi_kwargs = {
            'top_left': top_left, 
            'bot_right': bot_right, 
            'patch_size': patch_size, # (patch_val, patch_val)
            'step_size': step_size,   # Calculated based on overlap
            'custom_downsample': patch_args.custom_downsample,
            'level': patch_args.patch_level,
            'use_center_shift': heatmap_args.use_center_shift
        }

        # Save the "blockmap" (raw heatmap without blending, useful for debugging)
        blockmap_tiff_save_name = '{}_blockmap.tiff'.format(slide_id)
        blockmap_tiff_path = os.path.join(r_slide_save_dir, blockmap_tiff_save_name)
        if not os.path.isfile(blockmap_tiff_path):
            # Use scores and coordinates already loaded from block_map_h5_path
            heatmap_block = drawHeatmap_for_tia(
                scores_for_heatmap, coords_for_heatmap, slide_path, wsi_object=wsi_object, 
                cmap=heatmap_args.cmap, alpha=1.0, # alpha=1 to see only the score map
                use_holes=True, binarize=False, vis_level=0, blank_canvas=True, # blank_canvas=True
                thresh=-1, patch_size=vis_patch_size_for_heatmap, convert_to_percentiles=True, # patch_size here is at level 0
                top_left=top_left, bot_right=bot_right # Add ROI if present
            )
            heatmap_block.save(blockmap_tiff_path)
            del heatmap_block

        # Path to save/load final attention scores (might be different from block_map_h5_path if ROI)
        final_scores_h5_path = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5')

        if heatmap_args.use_ref_scores: # If True, use original scores (A_raw) as reference for percentiles
            ref_scores_for_final_heatmap = scores_for_heatmap # scores_for_heatmap are A_raw
        else:
            ref_scores_for_final_heatmap = None
        
        if heatmap_args.calc_heatmap: # If True, (re)calculate attention scores, potentially with ROI
            # print(f"Calculating final heatmap scores, saving to: {final_scores_h5_path}")
            final_scores_h5_path, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object, 
                img_transforms=img_transforms,
                clam_pred=Y_hats[0], model=model, 
                feature_extractor=feature_extractor, 
                batch_size=exp_args.batch_size, 
                **final_heatmap_wsi_kwargs, # Use kwargs with ROI if specified
                attn_save_path=final_scores_h5_path,
                ref_scores=ref_scores_for_final_heatmap # Pass ref_scores
            )

        # Load final scores for visualization
        if final_scores_h5_path and os.path.isfile(final_scores_h5_path):
            with h5py.File(final_scores_h5_path, 'r') as file:
                final_scores_to_draw = file['attention_scores'][:]
                final_coords_to_draw = file['coords'][:]
        elif block_map_h5_path and os.path.isfile(block_map_h5_path) and not heatmap_args.calc_heatmap:
            # Fallback to blockmap if calc_heatmap was False and final_scores_h5_path does not exist
            # print(f"Using scores from {block_map_h5_path} for final heatmap.")
            with h5py.File(block_map_h5_path, 'r') as file:
                final_scores_to_draw = file['attention_scores'][:]
                final_coords_to_draw = file['coords'][:]
        else:
            # print(f"Error: Scores for final heatmap not found at {final_scores_h5_path} or {block_map_h5_path}")
            continue
            
        # Arguments for final heatmap visualization
        heatmap_vis_args = {
            'convert_to_percentiles': True, # Default, can be overridden
            'vis_level': heatmap_args.vis_level, 
            'blur': heatmap_args.blur, 
            'custom_downsample': heatmap_args.custom_downsample
        }
        if heatmap_args.use_ref_scores: # If True, scores are already percentiles or raw as desired
            heatmap_vis_args['convert_to_percentiles'] = False 

        vis_level_for_heatmap_name = heatmap_args.vis_level
        if vis_level_for_heatmap_name < 0: # Se -1, usa un default o il livello della maschera
            vis_level_for_heatmap_name = vis_params_for_mask.get('vis_level',0) # Fallback a 0 se non definito

        heatmap_prod_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(
            slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
            int(heatmap_args.blur), int(heatmap_args.use_ref_scores), 
            int(heatmap_args.blank_canvas), float(heatmap_args.alpha), 
            int(vis_level_for_heatmap_name), # Usa il valore determinato
            int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext
        )
        heatmap_prod_path = os.path.join(p_slide_save_dir, heatmap_prod_save_name)

        if not os.path.isfile(heatmap_prod_path):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            heatmap_final_img = drawHeatmap_for_tia(
                final_scores_to_draw, final_coords_to_draw, slide_path, wsi_object=wsi_object,  
                cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
                binarize=heatmap_args.binarize, 
                blank_canvas=heatmap_args.blank_canvas,
                thresh=heatmap_args.binary_thresh,  
                patch_size=vis_patch_size_for_heatmap, # Patch size at level 0
                overlap=patch_args.overlap, 
                top_left=top_left, bot_right=bot_right, 
                **heatmap_vis_args # Pass visualization arguments
            )
            if heatmap_args.save_ext == 'jpg':
                heatmap_final_img.save(heatmap_prod_path, quality=100)
            else:
                heatmap_final_img.save(heatmap_prod_path)
        
        # Save the original WSI image if requested
        if heatmap_args.save_orig:
            # Usa heatmap_args.vis_level per l'immagine originale salvata, o un default
            orig_vis_level = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else 0 # Default a 0 se -1
            orig_save_name = f'{slide_id}_orig_lv{int(orig_vis_level)}.{heatmap_args.save_ext}'
            orig_save_path = os.path.join(p_slide_save_dir, orig_save_name)
            if not os.path.isfile(orig_save_path):
                img_orig = wsi_object.visWSI(vis_level=orig_vis_level, view_slide_only=True, 
                                             custom_downsample=heatmap_args.custom_downsample,
                                             top_left=top_left, bot_right=bot_right) # Add ROI if present
                if heatmap_args.save_ext == 'jpg':
                    img_orig.save(orig_save_path, quality=100)
                else:
                    img_orig.save(orig_save_path)

    # Save the final configuration used
    final_config_path = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config_final_run.yaml')
    with open(final_config_path, 'w') as outfile:
        yaml.dump(args_all, outfile, default_flow_style=False)