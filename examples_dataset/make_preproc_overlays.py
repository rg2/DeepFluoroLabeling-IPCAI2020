# This is an example script for overlaying ground truth segmentations
# and landmark points on 2D projections from a preprocessed dataset
#
# usage: python make_preproc_overlays.py <path to preproc. HDF5 file>
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

import torch
import torchvision.utils
import torchvision.transforms.functional as TF

# main function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay ground truth annotations in preprocessed file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_data_file_path', help='Path to preprocessed data file', type=str)
    parser.add_argument('--multi-class-seg', help='Use overlapping multiple-class segmentation', action='store_true')
    parser.add_argument('-a', '--alpha', help='Alpha blending coefficient of non-background label overlay. 1.0 --> non-background labels are opaque, 0.0 --> non-background labels are invisible.', type=float, default=0.35)

    args = parser.parse_args()

    do_multi_class = args.multi_class_seg
    
    alpha = args.alpha

    # variables used later for landmark overlay
    box_radius = None

    # open dataset file for reading
    f = h5.File(args.input_data_file_path, 'r')
  
    for spec_idx_str in f:
        spec_g = f[spec_idx_str]
       
        # handle the case of "land-names" which does not contain projection data
        if 'projs' not in spec_g:
            continue

        projs = torch.from_numpy(spec_g['projs'][:])
        lands = torch.from_numpy(spec_g['lands'][:])

        if do_multi_class:
            if not 'multi-segs' in spec_g:
                sys.stderr.write('ERROR: Multi-Class segmentation masks not present!\n')
                sys.exit(1)

            segs = torch.from_numpy(spec_g['multi-segs'][:])
        else:
            segs  = torch.from_numpy(spec_g['segs'][:])

        num_projs = projs.shape[0]

        proj_num_rows = projs.shape[1]
        proj_num_cols = projs.shape[2]
        
        num_lands = lands.shape[2]

        assert(num_projs == segs.shape[-4 if do_multi_class else -3])
        assert(proj_num_rows == segs.shape[-2])
        assert(proj_num_cols == segs.shape[-1])
        assert(num_projs == lands.shape[0])
        assert(lands.shape[1] == 2)
        
        if box_radius is None:
            box_radius = max(16 * (proj_num_rows / 1536.0), 3.0)

        # utility functions for overlaying landmark locations

        def get_box(x):
            return [(x[0] - box_radius, x[1] - box_radius),
                    (x[0] + box_radius, x[1] + box_radius)]

        def draw_land(draw_obj, x):
            draw_obj.ellipse(get_box(x), fill='yellow')

        proj_overlays = torch.zeros(num_projs, 3, proj_num_rows, proj_num_cols)

        for proj_idx in range(num_projs):
            cur_proj = projs[proj_idx,:,:]

            # normalize intensities into [0,1]
            cur_proj_min = cur_proj.min()
            cur_proj_max = cur_proj.max()
            cur_proj = (cur_proj - cur_proj_min) / (cur_proj_max - cur_proj_min)
            
            # convert the projection to RGB for overlaying the seg. in color
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
                cur_seg = segs[proj_idx,:,:,:]
                
                # loop over RGB
                for c in range(3):
                    cur_proj_c = cur_proj[c,:,:].clone()

                    cur_proj_c *= 1 - alpha
                    cur_proj_c += alpha * cur_seg[0,:,:] * cur_proj[c,:,:]

                    for l in range(1,7):
                        cur_proj_c += (alpha * label_colors[l-1][c]) * cur_seg[l,:,:]

                    cur_proj[c,:,:] = cur_proj_c
            else:
                cur_seg = segs[proj_idx,:,:]

                for l in range(1,7):
                    cur_label_color = label_colors[l - 1]
                    
                    if do_multi_class:
                        pass
                    else:
                        cur_label_idx = cur_seg == l

                        # loop over RGB
                        for c in range(3):
                            cur_proj_c = cur_proj[c,:,:]
                            
                            cur_proj_c[cur_label_idx] = ((1 - alpha) * cur_proj_c[cur_label_idx]) + (alpha * cur_label_color[c])
            
            cur_lands = lands[proj_idx,:,:]
                
            # Need a drawing context to overlay the landmarks
            pil = TF.to_pil_image(cur_proj)
            draw = ImageDraw.Draw(pil)

            for land_idx in range(num_lands):
                cur_land_col = cur_lands[0,land_idx]
                cur_land_row = cur_lands[1,land_idx]
                
                # only draw landmarks that are visible
                if (cur_land_col >= 0) and (cur_land_row >= 0) and \
                        (cur_land_col < proj_num_cols) and (cur_land_row < proj_num_cols):
                    draw_land(draw, [cur_land_col, cur_land_row])
            
            del draw
            
            cur_proj = TF.to_tensor(pil)

            proj_overlays[proj_idx,:,:,:] = cur_proj
        
        # write out the tiled image of overlays for this specimen.
        torchvision.utils.save_image(proj_overlays, '{}.png'.format(spec_idx_str), normalize=False)

