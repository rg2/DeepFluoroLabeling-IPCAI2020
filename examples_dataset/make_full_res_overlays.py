# This is an example script for overlaying ground truth segmentations
# and landmark points on 2D projections
#
# usage: python make_full_res_overlays.py <path to full-res HDF5 file>
# 
# outputs: in the current working directory, a PNG file for each
#          cadaveric specimen containing tiled overlays
#
# Copyright (C) 2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import sys
import argparse

import h5py as h5

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import torch
import torchvision.utils
import torchvision.transforms.functional as TF

# main function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay ground truth annotations from full-resolution data file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_data_file_path', help='Path to full-res data file', type=str)
    parser.add_argument('--multi-class-seg', help='Use overlapping multiple-class segmentation', action='store_true')
    parser.add_argument('-a', '--alpha', help='Alpha blending coefficient of non-background label overlay. 1.0 --> non-background labels are opaque, 0.0 --> non-background labels are invisible.', type=float, default=0.35)

    args = parser.parse_args()

    do_multi_class = args.multi_class_seg

    alpha = args.alpha
    
    # Since the projections are 1536x1536, a tiled image of ~100 projections
    # may be excessively large, downsample in this case
    #overlay_ds_factor = 1.0   # no downsampling
    overlay_ds_factor = 0.125 # downsample 8x in each 2D dim

    need_to_ds_overlay = abs(overlay_ds_factor - 1.0) > 0.001

    # utility functions for overlaying landmark locations

    def get_box(x):
        box_radius = 16
        return [(x[0] - box_radius, x[1] - box_radius),
                (x[0] + box_radius, x[1] + box_radius)]

    def draw_land(draw_obj, x):
        draw_obj.ellipse(get_box(x), fill='yellow')


    # try and get a nice font for the overlays
    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", 48)
    except:
        pass 

    # open dataset file for reading
    f = h5.File(sys.argv[1], 'r')

    # get the number of rows and columns in each projection
    # other information, such as extrinsic and intrinsic matrices
    # is also available
    proj_params_g = f['proj-params']
    proj_num_cols = proj_params_g['num-cols'][()]
    proj_num_rows = proj_params_g['num-rows'][()]

    # downsampled overlay dimensions
    ds_proj_num_cols = int(round(proj_num_cols * overlay_ds_factor)) if need_to_ds_overlay else proj_num_cols
    ds_proj_num_rows = int(round(proj_num_rows * overlay_ds_factor)) if need_to_ds_overlay else proj_num_rows

    # loop over the items in the file root
    for spec_id in f:
        # all items except proj-params are assumed to be groups
        # with data for a specific cadaver
        if spec_id != 'proj-params':
            # open up group with projections for this specimen
            projs_g = f['{}/projections'.format(spec_id)]

            num_projs = len(projs_g.keys())

            # we will store all of the projections in one tensor and then
            # save them to a tiled PNG later - using 3 channels since the
            # overlays will be RGB
            projs = torch.zeros(num_projs, 3, ds_proj_num_rows, ds_proj_num_cols)

            # loop over the projections
            for proj_idx in range(num_projs):
                cur_proj_g = projs_g['{:03d}'.format(proj_idx)]

                # read the current projection and convert into pytorch tensor
                cur_proj = torch.from_numpy(cur_proj_g['image/pixels'][:])
                
                # normalize intensities into [0,1]
                cur_proj_min = cur_proj.min()
                cur_proj_max = cur_proj.max()
                cur_proj = (cur_proj - cur_proj_min) / (cur_proj_max - cur_proj_min)
                
                if do_multi_class:
                    # read current ground truth multi-class segmentation weights
                    cur_seg = torch.from_numpy(cur_proj_g['gt-multi-seg'][:])
                else:
                    # read current ground truth 2D segmentation - dtype is uint8
                    cur_seg = torch.from_numpy(cur_proj_g['gt-seg/pixels'][:])
               
                # read the ground truth 2D landmarks, each landmark is stored
                # as a separate 2D vector in units of 2D pixels
                gt_lands_g = cur_proj_g['gt-landmarks']

                cur_lands = []
                
                # store the indices of the left/right femoral head landmarks, will use later to overlay some text
                land_idx_fhl = None
                land_idx_fhr = None
   
                land_idx = 0
                for land_name in gt_lands_g:
                    cur_land = gt_lands_g[land_name][:]
                    
                    # only keep landmarks that are visible in the projection
                    if (cur_land[0] >= 0) and (cur_land[1] >= 0) and \
                            (cur_land[0] < proj_num_cols) and (cur_land[1] < proj_num_cols):
                        cur_lands.append(cur_land)
                        
                        if land_name == 'FH-l':
                            land_idx_fhl = land_idx
                        elif land_name == 'FH-r':
                            land_idx_fhr = land_idx

                        land_idx += 1

                # when this flag is true, the projection needs to be rotated by 180 degrees
                # to make the patient appear "up" (superior in the top of image, inferior in the bottom)
                if cur_proj_g['rot-180-for-up'][()]:
                    # rotation of 180 deg. is equivalent to flipping columns, then flipping rows
                    cur_proj = torch.flip(torch.flip(cur_proj, [0]), [1])

                    if do_multi_class:
                        for i in range(cur_seg.shape[0]):
                            cur_seg[i,:,:] = torch.flip(torch.flip(cur_seg[i,:,:],  [0]), [1])
                    else:
                        cur_seg  = torch.flip(torch.flip(cur_seg,  [0]), [1])
                    
                    for cur_land in cur_lands:
                        cur_land[0] = proj_num_cols - 1 - cur_land[0]
                        cur_land[1] = proj_num_rows - 1 - cur_land[1]

                # Indicators that there is enough of a femur visible in the field of view
                # that we can use ground truth pose for any additional experiments, such as
                # evaluating another femur registration method
                left_femur_good_fov  = cur_proj_g['gt-poses/left-femur-good-fov'][()]
                right_femur_good_fov = cur_proj_g['gt-poses/right-femur-good-fov'][()]

                # Next the segmentation labels and landmark locations will be overlaid on the projections

                pil = TF.to_pil_image(cur_proj)
                pil = pil.convert('RGB')

                cur_proj = TF.to_tensor(pil)
                pil = None

                # alpha blending for segmentation overlay of pixels that are not background
                # 0 --> seg. not visible, only projection shows
                # 1 --> only seg. shows, proj. not visible in seg. regions

                label_colors = [ [0.0, 1.0, 0.0],  # green
                                 [1.0, 0.0, 0.0],  # red
                                 [0.0, 0.0, 1.0],  # blue
                                 [1.0, 1.0, 0.0],  # yellow
                                 [0.0, 1.0, 1.0],  # cyan
                                 [1.0, 0.5, 0.0]]  # orange
               
                # do alpha blending for each label
                
                if do_multi_class:
                    # loop over RGB
                    for c in range(3):
                        cur_proj_c = cur_proj[c,:,:].clone()

                        cur_proj_c *= 1 - alpha
                        cur_proj_c += alpha * cur_seg[0,:,:] * cur_proj[c,:,:]

                        for l in range(1,7):
                            cur_proj_c += (alpha * label_colors[l-1][c]) * cur_seg[l,:,:]

                        cur_proj[c,:,:] = cur_proj_c
                else:
                    for l in range(1,7):
                        cur_label_color = label_colors[l - 1]
                        
                        cur_label_idx = cur_seg == l

                        # loop over RGB
                        for c in range(3):
                            cur_proj_c = cur_proj[c,:,:]
                            
                            cur_proj_c[cur_label_idx] = ((1 - alpha) * cur_proj_c[cur_label_idx]) + (alpha * cur_label_color[c])

                # Need a drawing context to overlay the landmarks
                pil = TF.to_pil_image(cur_proj)
                draw = ImageDraw.Draw(pil)

                for cur_land in cur_lands:
                    draw_land(draw, cur_land)

                # Overlay some text indicating when their is sufficient FOV to ensure that a GT femur pose is valid

                if left_femur_good_fov:
                    draw.text(cur_lands[land_idx_fhl] if land_idx_fhl is not None else (0,0), 'L. Femur FOV OK', font=font)
                
                if right_femur_good_fov:
                    draw.text(cur_lands[land_idx_fhr] if land_idx_fhr is not None else (0,0), 'R. Femur FOV OK', font=font)

                del draw
                
                if need_to_ds_overlay:
                    pil = pil.resize((ds_proj_num_cols, ds_proj_num_rows), Image.BILINEAR)

                cur_proj = TF.to_tensor(pil)

                projs[proj_idx,:,:,:] = cur_proj

            # write out the tiled image of overlays for this specimen.
            torchvision.utils.save_image(projs, '{}.png'.format(spec_id), normalize=False)
    
