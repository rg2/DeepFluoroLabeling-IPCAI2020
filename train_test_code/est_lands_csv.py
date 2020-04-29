# Estimate landmark point locations using already estimated heatmaps
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys
import math
import time

import numpy as np

import torch

import h5py as h5

from dataset import *
from util import *

from ncc import ncc_2d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate landmark locations and write to CSV',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('heat_file_path', help='Path to dataset file containing labelings.', type=str)
    parser.add_argument('heats_group_path', help='H5 group path to heat maps', type=str)

    parser.add_argument('--out', help='output image path', type=str, default='yy_lands_est.csv')

    parser.add_argument('--pat', help='patient index', type=int)

    parser.add_argument('--use-seg', help='Path to segmentation dataset used to assist in detection', type=str, default='')
    
    parser.add_argument('--multi-class-seg', help='Use overlapping multiple-class segmentation', action='store_true')

    parser.add_argument('--global-ncc', help='Use global NCC on heatmap instead of local', action='store_true')

    parser.add_argument('--no-hdr', help='No CSV header', action='store_true')

    args = parser.parse_args()

    heat_file_path = args.heat_file_path
    heats_group_path = args.heats_group_path

    out_csv_path = args.out

    pat_ind = args.pat
    
    no_csv_hdr = args.no_hdr

    seg_ds_path = args.use_seg
    
    do_multi_class = args.multi_class_seg
    
    do_global_ncc = args.global_ncc

    land_names = get_land_names_from_dataset(heat_file_path)

    num_lands = len(land_names)
    
    seg_labels_to_use_for_lands = { 'FH-l'   : 5,
                                    'FH-r'   : 6,
                                    'GSN-l'  : 1,
                                    'GSN-r'  : 2,
                                    'IOF-l'  : 1,
                                    'IOF-r'  : 2,
                                    'MOF-l'  : 1,
                                    'MOF-r'  : 2,
                                    'SPS-l'  : 1,
                                    'SPS-r'  : 2,
                                    'IPS-l'  : 1,
                                    'IPS-r'  : 2,
                                    'ASIS-l' : 1,
                                    'ASIS-r' : 2,
                                    'PSIS-l' : 1,
                                    'PSIS-r' : 2,
                                    'PIIS-l' : 1,
                                    'PIIS-r' : 2 }

    multi_class_seg_masks_to_use_for_lands = { 'FH-l'   : [1,5],
                                               'FH-r'   : [2,6],
                                               'GSN-l'  : [1],
                                               'GSN-r'  : [2],
                                               'IOF-l'  : [1],
                                               'IOF-r'  : [2],
                                               'MOF-l'  : [1],
                                               'MOF-r'  : [2],
                                               'SPS-l'  : [1],
                                               'SPS-r'  : [2],
                                               'IPS-l'  : [1],
                                               'IPS-r'  : [2],
                                               'ASIS-l' : [1],
                                               'ASIS-r' : [2] }

    csv_out = open(out_csv_path, 'w')
    if not no_csv_hdr:
        csv_out.write('pat,proj,land,row,col,time\n')

    print('reading heatmaps...')
    f = h5.File(heat_file_path, 'r')
    heats = torch.from_numpy(f[heats_group_path][:])
    segs = None
    if seg_ds_path:
        segs = torch.from_numpy(f[seg_ds_path][:])
        
        if do_multi_class:
            assert(len(segs.shape) == 4)
        else:
            assert(len(segs.shape) == 3)

    f.close()

    landmark_local_template = 'global' if do_global_ncc else get_gaussian_2d_heatmap(25, 25, 2.5)

    print('detecting landmark locations...')
    for i in range(heats.shape[0]):
        #print('+-----------------------------------------+')
        for l in range(num_lands):
            start_time = time.time()

            cur_heat = heats[i,l,:,:]

            segs_to_use = None
            seg_labels_or_masks_to_use = None
            
            if segs is not None:
                if do_multi_class:
                    segs_to_use = segs[i,:,:,:]
                    seg_labels_or_masks_to_use = multi_class_seg_masks_to_use_for_lands[land_names[l]]
                else:
                    segs_to_use = segs[i,:,:]
                    seg_labels_or_masks_to_use = [seg_labels_to_use_for_lands[land_names[l]]]

            land_ind = est_land_from_heat(cur_heat,
                                          local_template=landmark_local_template,
                                          do_multi_class=do_multi_class,
                                          segs=segs_to_use,
                                          seg_labels_or_mask_inds=seg_labels_or_masks_to_use)

            if land_ind is None:
                # use -1 to indicate landmark not found
                land_ind = [-1, -1]
            
            stop_time = time.time()

            csv_line = '{},{},{},{},{},{:3f}'.format(pat_ind, i, l, land_ind[0], land_ind[1], stop_time - start_time)
            csv_out.write('{}\n'.format(csv_line))






