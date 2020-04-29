# Overlay estimated annotations (segmentation and landmark points)
# onto a projection. Optionally can overlay ground truth landmarks
# or omit the segmentation overlay.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys
import math

import torch
import torchvision.utils
import torchvision.transforms.functional as TF

import h5py as h5

from PIL import Image
from PIL import ImageDraw

from dataset import *
from overlay_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='overlay segs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ds_path', help='Path to dataset containing projections', type=str)
    parser.add_argument('seg_file', help='Path to H5 file with estimated segmentations and heatmaps', type=str)
    parser.add_argument('seg_group', help='Path within H5 file of estimated segmentations', type=str)
    parser.add_argument('pat_ind', help='patient index', type=int)
    parser.add_argument('proj_ind', help='proj', type=int)
    parser.add_argument('out_overlay', help='Path to output overlay image', type=str)

    parser.add_argument('--lands', help='overlay GT and est. landmark locations', action='store_true')
    
    parser.add_argument('--no-gt-lands', help='do not overlay GT landmarks', action='store_true')

    parser.add_argument('--no-seg', help='do not overlay est. seg.', action='store_true')

    parser.add_argument('--lands-csv', help='path to CSV file of estimated landmark locations', type=str)
    
    parser.add_argument('--num-classes', help='number of classes in segmentation', type=int, default=7)
    
    parser.add_argument('--multi-class-seg', help='Use overlapping multiple-class segmentation', action='store_true')
    
    parser.add_argument('-a', '--alpha', help='Alpha blending coefficient of non-background label overlay. 1.0 --> non-background labels are opaque, 0.0 --> non-background labels are invisible.', type=float, default=0.35)

    args = parser.parse_args()

    ds_path = args.ds_path
    
    seg_file_path = args.seg_file
    seg_g_path    = args.seg_group

    out_img_path = args.out_overlay

    pat_ind = args.pat_ind

    proj = args.proj_ind
    all_projs = proj < 0

    overlay_lands = args.lands

    no_gt_lands = args.no_gt_lands

    no_seg = args.no_seg

    num_seg_classes = args.num_classes

    alpha = args.alpha
    
    do_multi_class = args.multi_class_seg

    est_lands = { }

    if overlay_lands:
        all_est_lands = read_est_lands_from_csv(args.lands_csv)
        
        if pat_ind in all_est_lands:
            est_lands = all_est_lands[pat_ind]

    ds = get_dataset(ds_path, [pat_ind], num_classes=num_seg_classes)

    projs_to_use = range(len(ds)) if all_projs else [proj]
    
    dst_img = None

    for proj in projs_to_use:
        print('processing projection: {:03d}...'.format(proj))

        img = ds[proj][0]

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

        pil = TF.to_pil_image(img)
        pil = pil.convert('RGB')

        img = TF.to_tensor(pil)

        if not no_seg:
            f = h5.File(seg_file_path, 'r')
            segs = torch.from_numpy(f[seg_g_path][:])
            f.close()

            cur_seg = segs[proj,:,:]

            img = overlay_seg(img, cur_seg, alpha, do_multi_class, num_seg_classes)

        if overlay_lands:
            pil = TF.to_pil_image(img)
            
            draw = ImageDraw.Draw(pil)

            if not no_gt_lands:
                cur_gt_lands = ds[proj][2]

                for l in range(cur_gt_lands.shape[-1]):
                    cur_land = cur_gt_lands[:,l]

                    if math.isfinite(cur_land[0]) and math.isfinite(cur_land[1]):
                        draw_gt_land(draw, cur_land)
            
            for (l, cur_land) in est_lands[proj].items():
                draw_est_land(draw, cur_land)

            del draw

            img = TF.to_tensor(pil)
    
        if all_projs:
            if dst_img is None:
                dst_img = torch.zeros(len(ds), 3, img.shape[-2], img.shape[-1])
            dst_img[proj,:,:,:] = img
        else:
            dst_img = img

    torchvision.utils.save_image(dst_img, out_img_path, normalize=False)


