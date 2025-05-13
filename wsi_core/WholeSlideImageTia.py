import cv2
import os
from wsi_core.WholeSlideImage import WholeSlideImage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tiatoolbox.wsicore.wsireader import WSIReader
from utils.file_utils import load_pkl, save_pkl 
from wsi_core.wsi_utils import to_percentiles, screen_coords, save_hdf5 
import math
import multiprocessing as mp 

Image.MAX_IMAGE_PIXELS = None 

class WholeSlideImageTia(WholeSlideImage):
    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        try:
            self.wsi = WSIReader.open(path) #initialize WSIReader with tiatoolbox library
        except Exception as e:
            raise IOError(f"Could not open WSI {path} with Tiatoolbox: {e}")

        self.level_downsamples = self._init_level_downsamples()
        
        
        self.contours_tissue = None
        self.holes_tissue = None 

    def _init_level_dimensions(self):
        if self.wsi and hasattr(self.wsi, 'info') and hasattr(self.wsi.info, 'level_dimensions'):
            return self.wsi.info.level_dimensions
        return []

    def _init_level_downsamples(self):
        processed_downsamples = []
        if self.wsi and hasattr(self.wsi, 'info') and hasattr(self.wsi.info, 'level_downsamples'):
            for ds_info in self.wsi.info.level_downsamples:
                if isinstance(ds_info, (tuple, list)):
                    processed_downsamples.append(tuple(float(d) for d in ds_info))
                else: 
                    processed_downsamples.append((float(ds_info), float(ds_info)))
        return processed_downsamples
    
    def getOpenSlide(self): 
        return self.wsi

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale): # contours is a list of lists
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes_list] for holes_list in contours]

    @staticmethod
    def isInHoles(holes, pt, patch_size): # pt è (x,y)
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return True #point in a hole
        return False # not in a hole

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt): # 
            if holes is not None and holes: # if there are holes
                return not WholeSlideImageTia.isInHoles(holes, pt, patch_size) # true if not in holes
            else:
                return True 
        return False # not in contour

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImageTia.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    def getOpenSlide(self): 
        return self.wsi

    def segmentTissue(self, 
                        mask_resolution={'resolution': 1.25, 'units': 'power'}, 
                        filter_params={'a_t':100, 'a_h': 16, 'max_n_holes': 10}, # Parametri di filtro CLAM originali
                        exclude_ids=[], 
                        keep_ids=[]):
        """
        Segment tissue using tiatoolbox.wsicore.wsireader.WSIReader.tissue_mask().
        The countours are obtained from the tissue mask and scaled to level 0.
        The filtering is done using the same parameters as CLAM.
        """
        # print(f"Segmenting WSI with Tiatoolbox: {self.name}")
        
        try:
            mask_reader_obj = self.wsi.tissue_mask(
                method='otsu', # 'otsu' or 'threshold'
                resolution=mask_resolution['resolution'],
                units=mask_resolution['units']
            ) #obtain mask reader object (VirtualWSIReader)
            mask_width, mask_height = mask_reader_obj.slide_dimensions(resolution=0, units='level')
            
            raw_mask_data = mask_reader_obj.read_region(
                location=(0,0), level=0, size=(mask_width, mask_height)
            ) #return a np.array

            if isinstance(raw_mask_data, Image.Image):
                binary_mask_np = np.array(raw_mask_data.convert('L'))
            elif isinstance(raw_mask_data, np.ndarray):
                binary_mask_np = raw_mask_data
                if binary_mask_np.ndim == 3: # if it has 3 channels squeeze the last one
                    if binary_mask_np.shape[2] == 1:
                         binary_mask_np = binary_mask_np.squeeze(axis=2)
                    else: # es. RGB
                         binary_mask_np = cv2.cvtColor(binary_mask_np, cv2.COLOR_RGB2GRAY) if binary_mask_np.shape[2] >=3 else binary_mask_np[:,:,0]
            else:
                raise TypeError(f"Unexpected mask data type from Tiatoolbox: {type(raw_mask_data)}")

            # check if the mask is already binary
            if binary_mask_np.dtype != np.uint8:
                if np.issubdtype(binary_mask_np.dtype, np.floating) and \
                   np.max(binary_mask_np) <= 1.0 and np.min(binary_mask_np) >= 0.0:
                    binary_mask_np = (binary_mask_np * 255)
                binary_mask_np = binary_mask_np.astype(np.uint8)
            
            
            scale_factors_mask_to_level0 = self.wsi.convert_resolution_units(
                input_res=mask_resolution['resolution'], 
                input_unit=mask_resolution['units'], 
                output_unit='baseline' # 'baseline' corrisponde al downsample rispetto al livello 0
            )
            if isinstance(scale_factors_mask_to_level0, (tuple, list)):
                scale_x, scale_y = scale_factors_mask_to_level0[0], scale_factors_mask_to_level0[1]
            else: # single value
                scale_x = scale_y = float(scale_factors_mask_to_level0)

            contours_on_mask, hierarchy_on_mask = cv2.findContours(
                binary_mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            ) #find countours in the mask
            
            scaled_tissue_contours = []
            scaled_holes_by_tissue = [] 

            if hierarchy_on_mask is not None and len(contours_on_mask) > 0:
                hierarchy_on_mask = np.squeeze(hierarchy_on_mask, axis=(0,)) 
                if hierarchy_on_mask.ndim == 1: 
                    hierarchy_on_mask = np.array([hierarchy_on_mask])

                original_tissue_indices_map = {} 
                current_new_tissue_idx = 0

                for i, contour_mask_level in enumerate(contours_on_mask):
                    if hierarchy_on_mask[i][3] == -1: # external contour
                        scaled_cont = (contour_mask_level * np.array([scale_x, scale_y])).astype(np.int32)
                        scaled_tissue_contours.append(scaled_cont)
                        original_tissue_indices_map[i] = current_new_tissue_idx
                        scaled_holes_by_tissue.append([]) # initialize holes list
                        current_new_tissue_idx += 1
                
                for i, contour_mask_level in enumerate(contours_on_mask):
                    parent_idx_original = hierarchy_on_mask[i][3]
                    if parent_idx_original != -1 and parent_idx_original in original_tissue_indices_map:
                        new_tissue_idx_for_hole = original_tissue_indices_map[parent_idx_original]
                        scaled_hole = (contour_mask_level * np.array([scale_x, scale_y])).astype(np.int32)
                        scaled_holes_by_tissue[new_tissue_idx_for_hole].append(scaled_hole)
            
            self.contours_tissue = scaled_tissue_contours
            self.holes_tissue = scaled_holes_by_tissue
            
            def _filter_scaled_contours(scaled_contours_l0, scaled_holes_l0, filter_params_local_l0):
                filtered_contours = []
                filtered_holes_list = []

                for i, cont_l0 in enumerate(scaled_contours_l0):
                    a = cv2.contourArea(cont_l0)
                    current_holes_for_cont = scaled_holes_l0[i] if i < len(scaled_holes_l0) else []
                    hole_areas = [cv2.contourArea(h) for h in current_holes_for_cont]
                    a_effective = a - np.sum(hole_areas)

                    if a_effective == 0: continue
                    if a_effective > filter_params_local_l0.get('a_t', 100): # a_t is minimum area
                        filtered_contours.append(cont_l0)
                        
                        current_valid_holes = []
                        if current_holes_for_cont:
                            sorted_holes = sorted(current_holes_for_cont, key=cv2.contourArea, reverse=True)
                            sorted_holes = sorted_holes[:filter_params_local_l0.get('max_n_holes', 10)]
                            for hole in sorted_holes:
                                if cv2.contourArea(hole) > filter_params_local_l0.get('a_h', 16): # a_h is minimum hole area
                                    current_valid_holes.append(hole)
                        filtered_holes_list.append(current_valid_holes)
                
                return filtered_contours, filtered_holes_list
            
            self.contours_tissue, self.holes_tissue = _filter_scaled_contours(self.contours_tissue, self.holes_tissue, filter_params)

            if len(keep_ids) > 0:
                contour_indices_to_keep = set(keep_ids) - set(exclude_ids)
            else:
                contour_indices_to_keep = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)
            
            self.contours_tissue = [self.contours_tissue[i] for i in sorted(list(contour_indices_to_keep)) if i < len(self.contours_tissue)]
            self.holes_tissue = [self.holes_tissue[i] for i in sorted(list(contour_indices_to_keep)) if i < len(self.holes_tissue)] # Assumendo che l'ordine corrisponda

            # print(f"Tiatoolbox segmentation complete: {len(self.contours_tissue)} tissue contours found.")

        except Exception as e:
            # print(f"Error during Tiatoolbox segmentation for {self.name}: {e}")
            # import traceback
            # traceback.print_exc()
            self.contours_tissue = []
            self.holes_tissue = []

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
                    line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, 
                    view_slide_only=False, number_contours=False, seg_display=True, annot_display=True):

        if vis_level >= len(self.level_downsamples): vis_level = len(self.level_downsamples) -1
        if vis_level < 0: vis_level = 0

        current_downsample_factors = self.level_downsamples[vis_level]
        scale = [1.0/current_downsample_factors[0], 1.0/current_downsample_factors[1]]
        
        read_location_level0: tuple[int, int]
        output_size_at_vis_level: tuple[int, int]

        if top_left is not None and bot_right is not None:
            top_left_coord = tuple(top_left)
            read_location_level0 = top_left_coord
            
            w_level0_roi = bot_right[0] - top_left[0]
            h_level0_roi = bot_right[1] - top_left[1]
            
            output_size_at_vis_level = (
                int(w_level0_roi / current_downsample_factors[0]),
                int(h_level0_roi / current_downsample_factors[1])
            )
        else:
            top_left_coord = (0,0) 
            read_location_level0 = (0,0) 
            output_size_at_vis_level = self.wsi.slide_dimensions(resolution=vis_level, units='level')

        if output_size_at_vis_level[0] <= 0 or output_size_at_vis_level[1] <= 0:
            return Image.new("RGB", (100, 100), (230, 230, 230))

        img = self.wsi.read_rect(location=read_location_level0,
            size=output_size_at_vis_level,
            resolution=vis_level,
            units='level'
        )

        if not view_slide_only:
            offset_draw = tuple(-(np.array(top_left_coord) * scale).astype(int))
            
            effective_line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if effective_line_thickness < 1: effective_line_thickness = 1

            if self.contours_tissue is not None and seg_display:
                contours_tissue_vis = self.scaleContourDim(self.contours_tissue, scale)
                if not number_contours:
                    cv2.drawContours(img, contours_tissue_vis, -1, color, effective_line_thickness, lineType=cv2.LINE_AA, offset=offset_draw)
                else: 
                    for idx, cont_l0 in enumerate(self.contours_tissue):
                        if not cont_l0.size: continue
                        contour_vis_single = np.array(self.scaleContourDim([cont_l0], scale)[0])
                        contour_for_moment = contour_vis_single + offset_draw 
                        
                        try:
                            M = cv2.moments(contour_for_moment)
                            cX = int(M["m10"] / (M["m00"] + 1e-9))
                            cY = int(M["m01"] / (M["m00"] + 1e-9))
                            cv2.drawContours(img, [contour_vis_single], -1, color, effective_line_thickness, lineType=cv2.LINE_AA, offset=offset_draw)
                            cv2.putText(img, "{}".format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 10)
                        except ZeroDivisionError:
                            cv2.drawContours(img, [contour_vis_single], -1, color, effective_line_thickness, lineType=cv2.LINE_AA, offset=offset_draw)

                if self.holes_tissue is not None: # self.holes_tissue è lista di liste
                    holes_tissue_vis = self.scaleHolesDim(self.holes_tissue, scale)
                    for holes_list_vis in holes_tissue_vis:
                        cv2.drawContours(img, holes_list_vis, -1, hole_color, effective_line_thickness, lineType=cv2.LINE_AA, offset=offset_draw)
            
            if self.contours_tumor is not None and annot_display:
                contours_tumor_vis = self.scaleContourDim(self.contours_tumor, scale)
                cv2.drawContours(img, contours_tumor_vis, -1, annot_color, effective_line_thickness, lineType=cv2.LINE_AA, offset=offset_draw)
        
        img = Image.fromarray(img)
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)), Image.Resampling.LANCZOS)

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)), Image.Resampling.LANCZOS)
       
        return img

    def get_seg_mask(self, region_size_at_vis_level, scale_level0_to_vis, use_holes=False, offset_level0=(0,0)):

        tissue_mask = np.full(np.flip(region_size_at_vis_level), 0, dtype=np.uint8) # (h, w) per OpenCV
        
        if self.contours_tissue is None or not self.contours_tissue:
            return tissue_mask.astype(bool)

        contours_tissue_vis = self.scaleContourDim(self.contours_tissue, scale_level0_to_vis)
        draw_offset_vis = tuple(-(np.array(offset_level0) * scale_level0_to_vis).astype(np.int32))

        if self.holes_tissue and len(self.contours_tissue) == len(self.holes_tissue):
            combined = sorted(zip(contours_tissue_vis, self.scaleHolesDim(self.holes_tissue, scale_level0_to_vis)), 
                              key=lambda x: cv2.contourArea(x[0]), reverse=True)
            contours_tissue_sorted_vis = [item[0] for item in combined]
            holes_tissue_sorted_vis = [item[1] for item in combined]
        else: 
            contours_tissue_sorted_vis = contours_tissue_vis
            holes_tissue_sorted_vis = self.scaleHolesDim(self.holes_tissue, scale_level0_to_vis) if self.holes_tissue else []


        for idx, cont_vis in enumerate(contours_tissue_sorted_vis):
            cv2.drawContours(image=tissue_mask, contours=[cont_vis], contourIdx=-1, color=(1), offset=draw_offset_vis, thickness=cv2.FILLED)
            if use_holes and idx < len(holes_tissue_sorted_vis):
                cv2.drawContours(image=tissue_mask, contours=holes_tissue_sorted_vis[idx], contourIdx=-1, color=(0), offset=draw_offset_vis, thickness=cv2.FILLED)
                
        return tissue_mask.astype(bool)

    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(224, 224), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        if vis_level >= len(self.level_downsamples): vis_level = len(self.level_downsamples) -1
        if vis_level < 0: vis_level = 0

        current_downsample_factors = self.level_downsamples[vis_level]
        scale = np.array([1.0/current_downsample_factors[0], 1.0/current_downsample_factors[1]])
                
        if len(scores.shape) == 2: scores = scores.flatten()

        threshold_val = 0.0
        if binarize:
            threshold_val = thresh if thresh >= 0 else (1.0/len(scores) if len(scores) > 0 else 0.5)
        
        coords_level0 = np.copy(coords) # coords are already in level 0
        if top_left is not None and bot_right is not None:
            scores, coords_level0 = screen_coords(scores, coords_level0, top_left, bot_right)
            coords_relative_to_roi_level0 = coords_level0 - np.array(top_left)
            
            w_vis = int((bot_right[0] - top_left[0]) * scale[0])
            h_vis = int((bot_right[1] - top_left[1]) * scale[1])
            region_size_vis = (w_vis, h_vis)
            read_location_level0 = tuple(top_left)
        else:
            if vis_level >= len(self.level_dim): vis_level = len(self.level_dim) -1
            region_size_vis = self.level_dim[vis_level]
            w_vis, h_vis = region_size_vis
            coords_relative_to_roi_level0 = coords_level0 # (0,0) della WSI
            read_location_level0 = (0,0)

        if w_vis <= 0 or h_vis <= 0: return Image.new('RGB', (100,100), (230,230,230))

        coords_vis_roi_relative = np.ceil(coords_relative_to_roi_level0 * scale).astype(int)
        patch_size_vis = np.ceil(np.array(patch_size) * scale).astype(int)

        if convert_to_percentiles: scores = to_percentiles(scores)
        if not binarize and not convert_to_percentiles : scores /= 100.0 

        overlay_scores_vis = np.full(np.flip(region_size_vis), 0.0, dtype=float)
        counter_vis = np.full(np.flip(region_size_vis), 0, dtype=np.uint16)    
        
        for i in range(len(coords_vis_roi_relative)):
            s = scores[i]
            c_vis = coords_vis_roi_relative[i]
            if s >= threshold_val:
                if binarize: s = 1.0
                x_start, y_start = c_vis[0], c_vis[1]
                x_end, y_end = min(x_start + patch_size_vis[0], w_vis), min(y_start + patch_size_vis[1], h_vis)
                if x_end > x_start and y_end > y_start: # make sure the patch is valid
                    overlay_scores_vis[y_start:y_end, x_start:x_end] += s
                    counter_vis[y_start:y_end, x_start:x_end] += 1
        
        valid_pixels_mask = counter_vis > 0
        overlay_scores_vis[valid_pixels_mask] = overlay_scores_vis[valid_pixels_mask] / counter_vis[valid_pixels_mask]
        if binarize: overlay_scores_vis[valid_pixels_mask] = np.around(overlay_scores_vis[valid_pixels_mask])
        
        if blur:
            blur_k_size = tuple((patch_size_vis * (1-overlap)).astype(int) * 2 +1)
            blur_k_size = (max(1, blur_k_size[0] // 2 * 2 + 1), max(1, blur_k_size[1] // 2 * 2 + 1))
            overlay_scores_vis = cv2.GaussianBlur(overlay_scores_vis, blur_k_size, 0)  

        # transform to vis level
        base_img = self.wsi.read_rect(location=read_location_level0, size=region_size_vis, resolution=vis_level, units='level')

        cmap_obj = plt.get_cmap(cmap) 
        heatmap_rgb = (cmap_obj(overlay_scores_vis)[..., :3] * 255).astype(np.uint8) # ignore alpha channel

        final_blended_img = base_img.copy() #copy before applying the heatmap
        if segment and self.contours_tissue:
            tissue_mask_vis = self.get_seg_mask(region_size_vis, scale, use_holes, offset_level0=read_location_level0)
            # apply the heatmap only where the tissue mask is True
            # Where there is tissue (tissue_mask_vis == True), apply the blend
            if alpha < 1.0:
                 final_blended_img[tissue_mask_vis] = cv2.addWeighted(
                    heatmap_rgb[tissue_mask_vis], alpha, 
                    base_img[tissue_mask_vis], 1 - alpha, 0)
            else: # alpha = 1.0, show the heatmap on the tissue
                 final_blended_img[tissue_mask_vis] = heatmap_rgb[tissue_mask_vis]
        else: 
            if alpha < 1.0:
                final_blended_img = cv2.addWeighted(heatmap_rgb, alpha, base_img, 1 - alpha, 0)
            else: # alpha = 1.0, show only the heatmap
                final_blended_img = heatmap_rgb

        # Resize finale se custom_downsample o max_size
        w_final, h_final = final_blended_img.size
        if custom_downsample > 1:
            final_blended_img = final_blended_img.resize((int(w_final/custom_downsample), int(h_final/custom_downsample)), Image.Resampling.LANCZOS)
        
        w_final, h_final = final_blended_img.size # update after custom_downsample
        if max_size is not None and (w_final > max_size or h_final > max_size):
            resizeFactor = max_size/w_final if w_final > h_final else max_size/h_final
        final_blended_img = final_blended_img.resize((int(w_final*resizeFactor), int(h_final*resizeFactor)), Image.Resampling.LANCZOS)
       
        return final_blended_img

    def saveSegmentation(self, mask_file_path):
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file_path, asset_dict) # save_pkl da CLAM utils

    def initSegmentation(self, mask_file_path):
        if not os.path.isfile(mask_file_path):
            # print(f"Warning: Segmentation file {mask_file_path} not found.")
            self.contours_tissue = []
            self.holes_tissue = []
            return

        asset_dict = load_pkl(mask_file_path) # load_pkl da CLAM utils
        self.contours_tissue = asset_dict.get('tissue', [])
        self.holes_tissue = asset_dict.get('holes', [])