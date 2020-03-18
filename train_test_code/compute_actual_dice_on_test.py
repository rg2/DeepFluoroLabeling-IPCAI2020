# Utility to compute the average dice coefficients between estimated
# and ground truth segmentations. This is the actual dice score, not
# the differentiable dice loss used during training.
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys

import torch

import h5py as h5

from dataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute actual dice coefficients between estimated segmentations and ground truth. Scores are written out in CSV format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ds_path', help='Path to dataset containing projections', type=str)
    parser.add_argument('seg_file', help='Path to H5 file with estimated segmentations', type=str)
    parser.add_argument('seg_group', help='Path within H5 file of estimated segmentations', type=str)
    parser.add_argument('csv_out', help='Path to output CSV file', type=str)
    parser.add_argument('pat_ind', help='patient index', type=int)
    parser.add_argument('--no-hdr', help='No CSV header', action='store_true')

    parser.add_argument('--num-classes', help='number of classes in segmentation', type=int, default=7)

    args = parser.parse_args()

    ds_path = args.ds_path

    seg_file_path = args.seg_file
    seg_g_path    = args.seg_group
    
    csv_out_path = args.csv_out

    pat_ind = args.pat_ind

    no_csv_hdr = args.no_hdr

    num_seg_classes = args.num_classes

    f = h5.File(ds_path, 'r')
    gt_segs = torch.from_numpy(f['{:02d}/segs'.format(pat_ind)][:])
    f.close()
    
    num_projs = gt_segs.shape[0]

    f = h5.File(seg_file_path, 'r')
    est_segs = torch.from_numpy(f[seg_g_path][:])
    f.close()

    assert(num_projs == est_segs.shape[0])

    csv_out = open(csv_out_path, 'w')
    if not no_csv_hdr:
        csv_out.write('pat,proj,label,dice\n')

    for proj in range(num_projs):
        # exclude BG
        for l in range(1,num_seg_classes):
            def create_seg_mask(S,l):
                M = torch.zeros_like(S)
                M[S == l] = 1
                return M
            
            cur_gt_seg  = create_seg_mask(gt_segs[proj,:,:], l)
            cur_est_seg = create_seg_mask(est_segs[proj,:,:], l)
            
            inter_seg = cur_est_seg * cur_gt_seg

            gt_sum = float(torch.sum(cur_gt_seg).item())

            est_sum = float(torch.sum(cur_est_seg).item())

            inter_sum = float(torch.sum(inter_seg).item())

            tot_sum = est_sum + gt_sum

            d = 1.0

            if tot_sum > 0.1:
                d = (2.0 * inter_sum) / tot_sum
            else:
                assert(abs(inter_sum) < 1.0e-8)
            
            assert((-1.0e-8 < d) and (d < (1 + 1.0e-8)))

            csv_out.write(('{},{},{},{:.2f}\n'.format(pat_ind, proj, l, d)))

    csv_out.flush()
    csv_out.close()


