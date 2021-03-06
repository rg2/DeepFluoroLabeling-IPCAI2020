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

from dataset import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='overlay estimated heat maps for a specific projection and landmark',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ds_path', help='Path to dataset containing projections', type=str)
    parser.add_argument('seg_file', help='Path to H5 file with estimated segmentations and heatmaps', type=str)
    parser.add_argument('seg_group', help='Path within H5 file of estimated heatmaps', type=str)
    parser.add_argument('pat_ind', help='patient index', type=int)
    parser.add_argument('proj_ind', help='proj', type=int)
    parser.add_argument('land_ind', help='landmark index', type=int)
    parser.add_argument('out_overlay', help='Path to output overlay image', type=str)
    
    parser.add_argument('--num-classes', help='number of classes in segmentation', type=int, default=7)

    args = parser.parse_args()

    ds_path = args.ds_path
    
    seg_file_path = args.seg_file
    seg_g_path    = args.seg_group

    out_img_path = args.out_overlay

    pat_ind = args.pat_ind

    proj = args.proj_ind
    
    land_idx = args.land_ind

    num_seg_classes = args.num_classes

    ds = get_dataset(ds_path, [pat_ind], num_classes=num_seg_classes)

    img = ds[proj][0]

    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)


    pil = TF.to_pil_image(img)
    pil = pil.convert('RGB')

    img = TF.to_tensor(pil)
    
    f = h5.File(seg_file_path, 'r')
    est_heats = torch.from_numpy(f[seg_g_path][:])
    f.close()

    heat_base_color = [0.0, 1.0, 0.0]

    heat = est_heats[proj,land_idx,:,:]
    
    heat_min = heat.min()
    heat_max = heat.max()
    heat_min_minus_max = heat_max - heat_min

    heat = heat - heat_min
    if heat_min_minus_max > 1.0e-3:
        heat /= heat_min_minus_max

    for c in range(3):
        img[c,:,:] = ((1 - heat) * img[c,:,:]) + (heat * heat_base_color[c])

    torchvision.utils.save_image(img, out_img_path, normalize=False)



