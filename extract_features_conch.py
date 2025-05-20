import numpy as np
import torch
from PIL import Image
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
# from models import get_encoder # Rimosso perché usiamo CONCH direttamente
from conch.open_clip_custom import create_model_from_pretrained # Aggiunto per CONCH
from utils.file_utils import save_hdf5
from typing import Callable
# import pandas as pd # Già importato in tiatoolbox implicitamente o non strettamente necessario qui
import os
import argparse

from tqdm import tqdm
import glob

def _get_fallback_empty_data(model: torch.nn.Module, patch_size: int, device: str, error_context: str = ""):
    """
    Tries to infer feature_dim from the model and returns empty feature and coordinate arrays.
    Uses a fallback dimension if inference fails.
    """
    feature_dim = 768  # Default fallback
    try:
        dummy_input = torch.randn(1, 3, patch_size, patch_size).to(device)
        with torch.no_grad():
            # Assumendo che model.encode_image sia un metodo valido come da API CONCH
            feature_dim = model.encode_image(dummy_input, proj_contrast=False, normalize=False).shape[-1] # type: ignore
    except Exception as model_e:
        print(f"Could not infer feature_dim from CONCH model{error_context}: {model_e}. Using fallback dim {feature_dim}.")
    return np.array([]).reshape(0, feature_dim), np.array([]).reshape(0, 2)

def extract_features_from_wsi(
    wsi_path: str,
    model: torch.nn.Module,
    transform: Callable, # Sarà il preprocess di CONCH
    patch_size: int = 224,
    stride: int = 224,
    level: int = 0,
    min_mask_ratio: float = 0.5,
    device: str = "cpu"
):
    """
    Extracts features from a WSI using a pre-trained CONCH model.
    Args:
        wsi_path (str): Path to the WSI file.
        model (torch.nn.Module): Pre-trained CONCH model for feature extraction.
        transform (callable): Transformation function (preprocess) from CONCH to apply to patches.
        patch_size (int): Size of the patches to extract.
        stride (int): Stride for sliding window extraction.
        level (int): Level of the WSI to use for extraction.
        min_mask_ratio (float): Minimum mask ratio for valid patches.
        device (str): Device to use for computation ("cpu" or "cuda").
    """
    feats, coords_list = [], []
    extractor = None
    num_locations = 0

    print("Attempting to create patch extractor...")
    try:
        current_min_mask_ratio = min_mask_ratio
        extractor = SlidingWindowPatchExtractor(
            input_img=wsi_path,
            patch_size=(patch_size, patch_size),
            stride=(stride, stride),
            input_mask="otsu", # o un'altra maschera se disponibile/preferita
            min_mask_ratio=current_min_mask_ratio,
            units="level",
            resolution=level,
        )
        print("Extractor created successfully.")

        num_locations = len(extractor.locations_df) if hasattr(extractor, 'locations_df') and extractor.locations_df is not None else 0
        print(f"Extractor found {num_locations} valid patch locations after filtering with min_mask_ratio={current_min_mask_ratio}.")

    except Exception as e:
        print(f"ERROR creating SlidingWindowPatchExtractor for {os.path.basename(wsi_path)}: {e}")
        return _get_fallback_empty_data(model, patch_size, device, error_context=" (after extractor error)")

    if num_locations == 0:
        print(f"No valid patch locations found for {os.path.basename(wsi_path)} with min_mask_ratio={current_min_mask_ratio}.")
        return _get_fallback_empty_data(model, patch_size, device, error_context=" (num_locations is 0)")

    processed_patches = 0
    for i, img_np in enumerate(extractor): # extractor itera direttamente sulle patch numpy
        try:
            coord_row = extractor.locations_df.iloc[i]
            x = coord_row['x']
            y = coord_row['y']

            if not isinstance(img_np, np.ndarray):
                 print(f"Warning: Item {i} from extractor is not a numpy array (type: {type(img_np)}). Skipping.")
                 continue

            img_pil = Image.fromarray(img_np) # CONCH preprocess si aspetta PIL Image
            
            # apply CONCH transform and move to device
            t = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                # Usa encode_image con proj_contrast=False e normalize=False per features grezze
                f = model.encode_image(t, proj_contrast=False, normalize=False).cpu().numpy().squeeze(0)

            feats.append(f)
            coords_list.append((int(x), int(y)))
            processed_patches += 1

        except IndexError:
             print(f"Error: Index {i} out of bounds for locations_df (size: {len(extractor.locations_df)}). Stopping.")
             break
        except KeyError as e:
             print(f"Error: Column {e} not found in locations_df. Columns: {extractor.locations_df.columns}. Stopping.")
             break
        except Exception as e:
            coord_str = f" at coords approx ({extractor.locations_df.iloc[i]['x']}, {extractor.locations_df.iloc[i]['y']})" if 'x' in locals() and 'y' in locals() else ""
            print(f"Error processing patch {i}{coord_str} in {os.path.basename(wsi_path)}: {e}")
            continue

    if not feats:
        print(f"Warning: No features were successfully extracted for {os.path.basename(wsi_path)}.")
        return _get_fallback_empty_data(model, patch_size, device, error_context=" (no feats extracted)")
    
    try:
        final_feats = np.stack(feats, axis=0).astype(np.float32)
        final_coords = np.array(coords_list, dtype=np.int32)
        return final_feats, final_coords
    except ValueError as e:
        print(f"Error stacking features/coords for {os.path.basename(wsi_path)}: {e}")
        return _get_fallback_empty_data(model, patch_size, device, error_context=" (stacking error)")


def save_patches_h5( # Questa funzione sembra non essere usata nello script originale per salvare features, ma per patch grezze. La lascio se serve per altri scopi.
    out_path: str,
    patch_imgs: np.ndarray, # Questo suggerisce che salva immagini, non features
    coords: np.ndarray
):
    """
    Save patches to HDF5 file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    asset = {"imgs": patch_imgs.astype(np.uint8), "coords": coords}
    save_hdf5(out_path, asset, mode="w")


def process_wsi_directory(
    wsi_dir: str,
    output_dir: str,
    conch_model_cfg: str, # Nuovo argomento
    conch_checkpoint_path: str, # Nuovo argomento
    patch_size: int = 224,
    stride: int = 224,
    level: int = 0,
    min_mask_ratio: float = 0.5,
    device: str = "cpu",
    wsi_ext: str = ".tiff"
):
    """
    Processes all WSI files in a directory by extracting features and coordinates using CONCH.
    Loads the CONCH model only once.
    Saves features to .pt files and coordinates to separate .h5 files.
    Skips processing if both output .pt and .h5 files already exist.
    """
    print(f"Starting processing for directory: {wsi_dir}")
    print(f"Using device: {device}")
    print(f"Output will be saved in: {output_dir}")

    # --- Load CONCH model and transformation once ---
    print(f"Loading CONCH model '{conch_model_cfg}' from checkpoint '{conch_checkpoint_path}'...")
    try:
        # Usa force_img_size per adattare la patch_size
        model, transform = create_model_from_pretrained(
            model_cfg=conch_model_cfg, 
            checkpoint_path=conch_checkpoint_path,
            device=device, # Passa il device direttamente se create_model_from_pretrained lo supporta
            force_image_size=patch_size, # Passa patch_size a CONCH
            return_transform=True, # Assicura che venga restituito il preprocess
        ) # type: ignore
        model = model.to(device).eval() # Assicura che sia sul device corretto e in eval mode
        print("CONCH Model loaded successfully.")
    except Exception as e:
        print(f"!!!!!!!! FATAL ERROR: Could not load CONCH model '{conch_model_cfg}': {e}")
        import traceback
        traceback.print_exc()
        return
    # -------------------------------------------------

    # fetch all WSI files in the directory
    wsi_files = glob.glob(os.path.join(wsi_dir, f"*{wsi_ext}"))
    if not wsi_files:
        print(f"No WSI files found with extension '{wsi_ext}' in directory '{wsi_dir}'.")
        return

    print(f"Found {len(wsi_files)} WSI files to process.")

    # define the output directories
    pt_output_dir = os.path.join(output_dir, "pt_files")
    h5_output_dir = os.path.join(output_dir, "h5_files")
    os.makedirs(pt_output_dir, exist_ok=True)
    os.makedirs(h5_output_dir, exist_ok=True)

    # iterates on the WSI files and extracts features
    for wsi_file_path in tqdm(wsi_files, desc="Processing WSIs"):
        slide_id = os.path.splitext(os.path.basename(wsi_file_path))[0]
        
        pt_output_path = os.path.join(pt_output_dir, f"{slide_id}.pt")
        h5_output_path = os.path.join(h5_output_dir, f"{slide_id}.h5")

        if os.path.isfile(pt_output_path) and os.path.isfile(h5_output_path):
            # print(f"Skipping {slide_id}: Output files .pt and .h5 already exist.") 
            continue

        print(f"\nProcessing slide: {slide_id}")
        try:
            feats, coords = extract_features_from_wsi(
                wsi_path=wsi_file_path,
                model=model,
                transform=transform, # Questa è la preprocess di CONCH
                patch_size=patch_size,
                stride=stride,
                level=level,
                min_mask_ratio=min_mask_ratio,
                device=device
            )
        except Exception as e:
            print(f"!!!!!!!! ERROR during feature extraction for {slide_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # save the features in .pt
        if feats.size > 0: # Controlla se feats non è vuoto
            try:
                features_tensor = torch.from_numpy(feats)
                torch.save(features_tensor, pt_output_path)
                print(f"Saved features for {slide_id} to {pt_output_path}")
            except Exception as e:
                print(f"Error saving features to .pt file for {slide_id}: {e}")
        else:
            print(f"No features extracted for {slide_id}, skipping .pt file saving.")


        if coords.size > 0: # Controlla se coords non è vuoto
            try:
                asset_dict = {"coords": coords.astype(np.int32)}
                save_hdf5(h5_output_path, asset_dict, mode="w")
                print(f"Saved coordinates for {slide_id} to {h5_output_path}")
            except Exception as e:
                print(f"Error saving coordinates to .h5 file for {slide_id}: {e}")
        else:
            print(f"No coordinates extracted for {slide_id}, skipping .h5 file saving.")


    print("\nFinished processing all WSI files in the directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from WSIs in a directory using TIAToolbox and a CONCH model.')

    parser.add_argument('--wsi_dir', type=str, required=True,
                        help='Directory containing the WSI files (e.g., .tiff, .svs).')
    parser.add_argument('--output_dir', type=str, default='./output_conch_features/',
                        help='Main output directory where pt_files and h5_files subdirectories will be created.')
    
    # Argomenti specifici per CONCH
    parser.add_argument('--conch_model_cfg', type=str, required=True, 
                        help="CONCH model configuration (e.g., 'conch_ViT-B-16').")
    parser.add_argument('--conch_checkpoint_path', type=str, required=True,
                        help="Path to the CONCH model checkpoint (.bin file).")

    parser.add_argument('--patch_size', type=int, default=224, # CONCH di default usa 448, ma può essere forzato
                        help='Size of the patches to extract and input to CONCH (e.g., 224, 336, 448). Passed to force_img_size.')
    parser.add_argument('--stride', type=int, default=224,
                        help='Stride for the sliding window patch extractor.')
    parser.add_argument('--level', type=int, default=0,
                        help='WSI level to use for patch extraction.')
    parser.add_argument('--min_mask_ratio', type=float, default=0.1, # Abbassato per test, potrebbe necessitare tuning
                        help='Minimum tissue mask ratio required for a patch to be processed.')
    parser.add_argument('--wsi_ext', type=str, default='.tiff',
                        help='Extension of the WSI files to process (e.g., .tiff, .svs).')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    process_wsi_directory(
        wsi_dir=args.wsi_dir,
        output_dir=args.output_dir,
        conch_model_cfg=args.conch_model_cfg,
        conch_checkpoint_path=args.conch_checkpoint_path,
        patch_size=args.patch_size,
        stride=args.stride,
        level=args.level,
        min_mask_ratio=args.min_mask_ratio,
        device=device,
        wsi_ext=args.wsi_ext
    )