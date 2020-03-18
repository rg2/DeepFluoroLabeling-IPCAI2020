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

    args = parser.parse_args()

    ds_path = args.ds_path
    
    seg_file_path = args.seg_file
    seg_g_path    = args.seg_group

    out_img_path = args.out_overlay

    pat_ind = args.pat_ind

    proj = args.proj_ind

    overlay_lands = args.lands

    no_gt_lands = args.no_gt_lands

    no_seg = args.no_seg

    num_seg_classes = args.num_classes

    est_lands = { }

    if overlay_lands:
        est_lands_csv_path = args.lands_csv
        csv_lines = open(est_lands_csv_path, 'r').readlines()[1:]
        
        for csv_line in csv_lines:
            toks = csv_line.strip().split(',')
            if (int(toks[0]) == pat_ind) and (int(toks[1]) == proj):
                land_row = int(toks[3])
                land_col = int(toks[4])

                if (land_row >= 0) and (land_col >= 0):
                    est_land_idx = int(toks[2])

                    assert(est_land_idx not in est_lands)

                    est_lands[est_land_idx] = (land_col, land_row)

    ds = get_dataset(ds_path, [pat_ind], num_classes=num_seg_classes)

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

        alpha = 0.35

        label_colors = [ [0.0, 1.0, 0.0],  # pelvis green
                         [1.0, 0.0, 0.0],  # left femur red
                         [0.0, 0.0, 1.0],  # right femur blue
                         [1.0, 1.0, 0.0],  # yellow
                         [0.0, 1.0, 1.0],  # cyan
                         [1.0, 0.5, 0.0],  # orange
                         [0.5, 0.0, 0.5]]  # purple

        for l in range(1,num_seg_classes):
            s_idx = cur_seg == l
            
            label_color = label_colors[l - 1]
            
            for c in range(3):
                img_c = img[c,:,:]

                img_c[s_idx] = ((1 - alpha) * img_c[s_idx]) + (alpha * label_color[c])

    if overlay_lands:
        pil = TF.to_pil_image(img)
        
        draw = ImageDraw.Draw(pil)

        def get_box(x, box_radius=2):
            return [(x[0] - box_radius, x[1] - box_radius),
                    (x[0] + box_radius, x[1] + box_radius)]

        def draw_gt_land(draw_obj, x):
            draw_obj.ellipse(get_box(x), fill='yellow')
        
        def draw_est_land(draw_obj, x):
            r = 6
            color_str = 'yellow'
            
            draw_obj.line([(x[0], x[1] + r), (x[0], x[1] - r)], fill=color_str)
            draw_obj.line([(x[0] - r, x[1]), (x[0] + r, x[1])], fill=color_str)

        if not no_gt_lands:
            cur_gt_lands = ds[proj][2]

            for l in range(cur_gt_lands.shape[-1]):
                cur_land = cur_gt_lands[:,l]

                if math.isfinite(cur_land[0]) and math.isfinite(cur_land[1]):
                    draw_gt_land(draw, cur_land)
        
        for (l, cur_land) in est_lands.items():
            draw_est_land(draw, cur_land)

        del draw

        img = TF.to_tensor(pil)

    torchvision.utils.save_image(img, out_img_path, normalize=False)


