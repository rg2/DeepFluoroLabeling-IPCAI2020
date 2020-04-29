# Overlay estimated heatmap for one landmark on a specific projection.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys

import torch
import torchvision.utils
import torchvision.transforms.functional as TF

import h5py as h5

from PIL import Image
from PIL import ImageDraw

from dataset import *
from util import *
from overlay_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='overlay estimated heat maps for a specific projection and landmark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ds_path', help='Path to dataset containing projections', type=str)
    parser.add_argument('seg_file', help='Path to H5 file with estimated segmentations and heatmaps', type=str)
    parser.add_argument('seg_group', help='Path within H5 file of estimated heatmaps', type=str)
    parser.add_argument('pat_ind', help='patient index', type=int)
    parser.add_argument('proj_ind', help='Projection index, negative value implies all projections', type=int)
    parser.add_argument('land_ind', help='landmark index, negative value implies all landmarks. Landmark overlays are tiled into one output image per projection', type=int)
    parser.add_argument('out_overlay', help='Path to output overlay image, should be a pattern for multiple projs (include a {})', type=str)
    
    parser.add_argument('--num-classes', help='number of classes in segmentation', type=int, default=7)

    parser.add_argument('--min-max', help='Min/Max normalize each heatmap to [0,1] for display', action='store_true')
    parser.add_argument('--exp-max', help='Max scale according to the expected peak value of the 2D Gaussian', action='store_true')
    parser.add_argument('--all-max', help='Max scale using maxium from all heatmaps of projection, shift to zero using individual heatmap minima.', action='store_true')
    parser.add_argument('--est-land', help='Estimate landmark location from heatmap and overlay', action='store_true')

    args = parser.parse_args()

    ds_path = args.ds_path
    
    seg_file_path = args.seg_file
    seg_g_path    = args.seg_group

    out_img_path = args.out_overlay

    pat_ind = args.pat_ind

    proj = args.proj_ind
    all_projs = proj < 0
    
    land_idx = args.land_ind
    all_lands = land_idx < 0

    num_seg_classes = args.num_classes

    do_min_max_norm  = args.min_max
    do_exp_max       = args.exp_max
    do_all_heats_max = args.all_max
    
    do_est_land = args.est_land

    ds = get_dataset(ds_path, [pat_ind], num_classes=num_seg_classes)
        
    f = h5.File(seg_file_path, 'r')
    est_heats = torch.from_numpy(f[seg_g_path][:])
    f.close()
    
    heat_base_color = [0.0, 1.0, 0.0]

    proj_inds_to_do = range(len(ds)) if all_projs else [proj]
   
    land_inds_to_do = range(est_heats.shape[1]) if all_lands else [land_idx]
        
    tile_overlay = None
    if all_lands:
        tile_overlay = torch.zeros(len(land_inds_to_do), 3, est_heats.shape[-2], est_heats.shape[-1])

    for proj in proj_inds_to_do:
        print('processing projection: {:03d}....'.format(proj))

        img = ds[proj][0]

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

        pil = TF.to_pil_image(img)
        pil = pil.convert('RGB')

        img = TF.to_tensor(pil)
       
        for land_idx in land_inds_to_do:
            heats_max = None

            heat = est_heats[proj,land_idx,:,:]

            # estimate landmark location
            land_est = None
            if do_est_land:
                #land_est = est_land_from_heat(heat)
                land_est = est_land_from_heat(heat,local_template='global')

            if do_min_max_norm:
                heat_min = heat.min()
                heat_max = heat.max()
                heat_min_minus_max = heat_max - heat_min
                
                heat = (heat - heat_min) / heat_min_minus_max
            elif do_exp_max:
                heat_map_sigma = 2.5
                heat *= 2 * math.pi * heat_map_sigma * heat_map_sigma
            elif do_all_heats_max:
                if heats_max is None:
                    heats_max = est_heats[proj,:,:,:].max()
                
                heat_min  = heat.min()

                heat = (heat - heat_min) / heats_max
            # else no normalization

            dst_img = tile_overlay[land_idx,:,:,:] if all_lands else img

            for c in range(3):
                dst_img[c,:,:] = ((1 - heat) * img[c,:,:]) + (heat * heat_base_color[c])
            
            if land_est is not None:
                pil = TF.to_pil_image(dst_img)
                draw = ImageDraw.Draw(pil)
                draw_est_land(draw, (land_est[1], land_est[0]))
                del draw

                dst_img[:,:,:] = TF.to_tensor(pil)
        
        cur_out_img_path = out_img_path
        if all_projs:
            cur_out_img_path = out_img_path.format('{:03d}'.format(proj))

        torchvision.utils.save_image(tile_overlay if all_lands else img, cur_out_img_path, normalize=False)

